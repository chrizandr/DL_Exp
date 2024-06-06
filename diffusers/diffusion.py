
"""
Vanilla PyTorch Implementation of 'Denoising Diffusion Probabilistic Models'
https://arxiv.org/pdf/2006.11239

Author: chrizandr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from einops.layers.torch import Rearrange
from torch.utils.data import Dataset
import os


class UNet(nn.Module):
    def __init__(self, embedding_shape):
        super().__init__()
        self.resnet_1 = ResnetBlock(in_size=3, out_size=64)
        self.down_1 = DownsampleBlock(
            embedding_shape, in_size=64, out_size=128)
        self.attention_1 = AttentionBlock(num_channels=128, heads=4, w=32, h=32)
        self.down_2 = DownsampleBlock(
            embedding_shape, in_size=128, out_size=256)
        self.attention_2 = AttentionBlock(num_channels=256, heads=4, w=16, h=16)
        self.down_3 = DownsampleBlock(
            embedding_shape, in_size=256, out_size=256)
        self.attention_3 = AttentionBlock(num_channels=256, heads=4, w=8, h=8)
        self.bottle_neck = nn.Sequential(
            ResnetBlock(in_size=256, out_size=512),
            ResnetBlock(in_size=512, out_size=512),
            ResnetBlock(in_size=512, out_size=256),
        )
        self.upsample_1 = UpsampleBlock(
            embedding_shape, in_size=256, out_size=128)
        self.attention_rev_1 = AttentionBlock(num_channels=128, heads=4, w=16, h=16)
        self.upsample_2 = UpsampleBlock(
            embedding_shape, in_size=128, out_size=64)
        self.attention_rev_2 = AttentionBlock(num_channels=64, heads=4, w=32, h=32)
        self.upsample_3 = UpsampleBlock(
            embedding_shape, in_size=64, out_size=64)
        self.attention_rev_3 = AttentionBlock(num_channels=64, heads=4, w=64, h=64)
        self.final_conv = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x, time_embedd):
        res1_out = self.resnet_1(x)
        out = self.down_1(res1_out, time_embedd)
        atten1_out = self.attention_1(out)
        out = self.down_2(atten1_out, time_embedd)
        atten2_out = self.attention_2(out)
        out = self.down_3(atten2_out, time_embedd)
        out = self.attention_3(out)

        out = self.bottle_neck(out)

        out = self.upsample_1(out, atten2_out, time_embedd)
        out = self.attention_rev_1(out)
        out = self.upsample_2(out, atten1_out, time_embedd)
        out = self.attention_rev_2(out)
        out = self.upsample_3(out, res1_out, time_embedd)
        out = self.attention_rev_3(out)
        out = self.final_conv(out)
        return out

class ResnetBlock(nn.Module):
    def __init__(self, in_size=3, out_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_size, out_channels=out_size,
                    kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=1, num_channels=out_size)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=out_size, out_channels=out_size,
                    kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=1, num_channels=out_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, embedding_shape, in_size=64, out_size=128):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.resnet1 = ResnetBlock(in_size=in_size, out_size=out_size)
        self.resnet2 = ResnetBlock(out_size=out_size, in_size=out_size)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_shape, out_size)
        )

    def forward(self, x, time_emedd):
        out = self.pool(x)
        out = self.resnet1(out)
        out = self.resnet2(out)
        time_out = self.time_mlp(time_emedd)
        time_out = time_out[(..., ) + (None, ) * 2]
        out = out + time_out
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, embedding_shape, in_size=64, out_size=128):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.resnet1 = ResnetBlock(in_size=2*in_size, out_size=out_size)
        self.resnet2 = ResnetBlock(out_size=out_size, in_size=out_size)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_shape, out_size)
        )

    def forward(self, x, res, time_emedd):
        out = self.upsample(x)
        out = torch.cat((out, res), dim=1)
        out = self.resnet1(out)
        out = self.resnet2(out)
        time_out = self.time_mlp(time_emedd)
        time_out = time_out[(..., ) + (None, ) * 2]
        out = out + time_out
        return out


class AttentionBlock(nn.Module):
    def __init__(self, num_channels=64, heads=4, w=64, h=64):
        super().__init__()
        self.rearrange = Rearrange('n c w h -> n (w h) c')
        self.reverse_rearrange = Rearrange('n (w h) c -> n c w h', w=w, h=h)
        self.lnorm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(heads=heads, dk=num_channels, dv=num_channels)
        self.lnorm2 = nn.LayerNorm(num_channels)
        self.linear1 = nn.Linear(num_channels, num_channels)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(num_channels, num_channels)

    def forward(self, x):
        reshaped_out = self.rearrange(x)
        norm_out = self.lnorm(reshaped_out)
        attention_out = self.attention(norm_out, norm_out, norm_out)
        out = attention_out + reshaped_out
        out = self.lnorm2(out)
        out = self.linear1(out)
        out = self.gelu(out)
        out = self.linear2(out)
        out = out + attention_out
        out = self.reverse_rearrange(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, dk, dv):
        super().__init__()
        self.attention_blocks = [Attention(
            dk, dv, heads) for i in range(heads)]

    def forward(self, keys, queries, values, masking=False):
        out = [x(keys, queries, values, masking)
               for x in self.attention_blocks]
        out = torch.concat(out, 2)  # B x N x (D/h) -> B x N x D
        return out


class Attention(nn.Module):
    def __init__(self, dk, dv, heads=1):
        super().__init__()
        self.WQ = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WK = torch.nn.Linear(dk, int(dk / heads), bias=False)
        self.WV = torch.nn.Linear(dv, int(dv / heads), bias=False)

    def forward(self, keys, queries, values, masking=False):
        Q_ = self.WQ(queries)
        K_ = self.WK(keys)
        K_ = K_.permute(0, 2, 1)

        y = torch.bmm(Q_, K_) / np.sqrt(keys.shape[1])
        if masking:
            key_shape = keys.shape[1]
            queries_shape = queries.shape[1]
            mask = torch.tensor([[0 if i <= j else -1 * float("inf")
                                for i in range(key_shape)] for j in range(queries_shape)])
            y = mask + y

        z = F.softmax(y, 1)
        output = torch.matmul(z, self.WV(values))
        return output


class DiffusionDataset(Dataset):
    def __init__(self, images_folder,
            embedding_dim,
            beta_start=1e-4,
            beta_end=0.02,
            num_steps=1000):

        super().__init__()
        self.img_files = [os.path.join(images_folder, x) for x in os.listdir(images_folder) if x.endswith(".png")]
        self.num_timesteps = num_steps
        self.pos_embeddings = DiffusionDataset.positional_encodings(
            embedding_dim, num_steps)

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])

        self.beta_range = torch.arange(beta_start, beta_end, step=(
            beta_end - beta_start) / num_steps)
        self.alpha_range = 1 - self.beta_range
        self.alpha_prods = torch.cumprod(self.alpha_range, dim=0)

    def load_image(self, index):
        image_path = self.img_files[index]
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_tensor = torch.tensor(image_array, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1)
        image = self.transform(image_tensor)
        return image

    @classmethod
    def positional_encodings(cls, embedding_dim, num_timesteps=1024):
        i = np.array(list(range(embedding_dim)))
        even_mask = (i % 2 == 0).astype(int)
        encodings = []
        power = 10000 ** (2 * i / embedding_dim)
        for pos in range(num_timesteps):
            angles = pos / power
            sins = np.sin(angles)
            coses = np.cos(angles)
            encoding = sins * even_mask + coses * (1-even_mask)
            encodings.append(encoding)
        return torch.tensor(np.array(encodings)).float()

    def forward_diffusion_one_step(self, img, timestep):
        alpha_t = self.alpha_prods[timestep]
        b = torch.sqrt(1-alpha_t)
        a = torch.sqrt(alpha_t)
        epsilon = torch.randn_like(img)
        x_t = a * img + b * epsilon
        return x_t, epsilon

    def forward_diffusion_iterative(self, img, timesteps):
        x_t = torch.clone(img)
        for i in range(timesteps):
            beta = self.beta_range[i]
            mean = torch.sqrt(1-beta) * x_t
            std = beta * torch.eye(img.shape[0], img.shape[1])
            x_t = torch.normal(mean, std)
        return x_t

    def __getitem__(self, index):
        image = self.load_image(index)
        timestep = torch.tensor(np.random.choice(self.num_timesteps))
        time_embedding = self.pos_embeddings[timestep, ::]
        diffused_image, epsilon = self.forward_diffusion_one_step(image, timestep)
        return diffused_image, epsilon, time_embedding, timestep


if __name__ == "__main__":
    embedding_shape = 1024
    out = torch.randn((2, 3, 64, 64))
    res = torch.randn((2, 64, 64, 64))
    time_embedd = torch.randn((1, 1024))
    dataset = DiffusionDataset("/Users/chris/DL_Exp/diffusers/images", embedding_shape)
    diffused_image, epsilon, time_embedding, timestep = dataset[0]
    model = UNet(embedding_shape)
    out = model(diffused_image.unsqueeze(0), time_embedding.unsqueeze(0))