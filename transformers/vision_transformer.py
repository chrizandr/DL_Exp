"""
Vanilla PyTorch Implementation of 'An Image is Worth 16X16 Words: Transformers for Image Recognition at scale'
https://arxiv.org/pdf/2010.11929

Author: chrizandr
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


CLS_TOKEN = torch.tensor([0])


class ClassEmbedding(nn.Module):
    def __init__(self, dmodel=128):
        super().__init__()
        self.class_embedding = nn.Embedding(1, dmodel)

    def forward(self, x):
        x = self.class_embedding(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes=100, dmodel=128, hidden_dim=2048, finetuning=False):
        super().__init__()
        if not finetuning:
            self.layers = nn.Sequential(
                    nn.Linear(dmodel, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.layers = nn.Linear(dmodel, num_classes)

    def forward(self, x):
        x = self.layers(x)
        return x


class MultiHeadSDPAttentionBlock(nn.Module):
    def __init__(self, heads, dk, dv):
        super().__init__()
        self.attention_blocks = [SDPAttentionBlock(
            dk, dv, heads) for i in range(heads)]

    def forward(self, keys, queries, values, masking=False):
        out = [
            x(keys, queries, values, masking)
            for x in self.attention_blocks
        ]
        out = torch.concat(out, 2)  # B x N x (D/h) -> B x N x D
        return out


class SDPAttentionBlock(nn.Module):
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
            num_tokens = keys.shape[1]
            mask = torch.tensor([[0 if i <= j else -1 * float("inf")
                                for i in range(num_tokens)] for j in range(num_tokens)])
            y = mask + y

        z = F.softmax(y, 1)
        output = torch.matmul(z, self.WV(values))
        return output


class NNBlock(nn.Module):
    def __init__(self, dv, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(dv, hidden_dim, bias=True)
        self.l2 = nn.Linear(hidden_dim, dv, bias=True)

    def forward(self, values):
        z1 = F.relu(self.l1(values))
        z2 = self.l2(z1)
        return z2


def positional_encodings(dk, context_length=2048):
    i = np.array(list(range(dk)))
    even_mask = (i % 2 == 0).astype(int)
    encodings = []
    power = 10000 ** (2 * i / dk)

    for pos in range(context_length):
        angles = pos / power
        sins = np.sin(angles)
        coses = np.cos(angles)
        encoding = sins * even_mask + coses * (1-even_mask)
        encodings.append(encoding)
    return torch.tensor(np.array(encodings)).float()


class EncoderBlock(nn.Module):
    def __init__(self, dk, dv, hidden_dim, heads):
        super().__init__()
        self.lnorm1 = nn.LayerNorm(dv)
        self.lnorm2 = nn.LayerNorm(dv)
        self.attention = MultiHeadSDPAttentionBlock(heads, dk, dv)
        self.ffn = NNBlock(dv, hidden_dim)

    def forward(self, keys, queries, values):
        z = self.attention(keys, queries, values)
        z = self.lnorm1(z + queries)
        y = self.ffn(z)
        y = self.lnorm2(y + z)
        return y


class Encoder(nn.Module):
    def __init__(self, num_blocks, dk, dv, hidden_dim, heads):
        super().__init__()
        self.blocks = [
            EncoderBlock(dk, dv, hidden_dim, heads)
            for _ in range(num_blocks)
        ]

    def forward(self, keys, queries, values):
        for block in self.blocks:
            out = block(keys, queries, values)
            keys, queries, values = out, out, out
        return values


class VisionTransformer(nn.Module):
    def __init__(self, context_length, dmodel, num_classes, hidden_dim, heads, img_dim=256, num_blocks=4, finetuning=False):
        super().__init__()
        self.context_length = context_length
        self.pos = positional_encodings(dmodel, context_length)
        self.image_projection = nn.Linear(img_dim, dmodel)
        self.class_embedding = ClassEmbedding(dmodel)
        self.encoder = Encoder(num_blocks, dmodel, dmodel, hidden_dim, heads)
        self.classification_head = ClassificationHead(num_classes, dmodel, hidden_dim, finetuning=finetuning)

    def gen_class_tokens(self, batch):
        batch_size = batch.shape[0]
        class_tokens = torch.tensor([CLS_TOKEN] * batch_size)
        class_embedding = self.class_embedding(class_tokens)
        return class_embedding.unsqueeze(1)

    def forward(self, image_vectors):
        assert len(image_vectors) < self.context_length
        class_embedding = self.gen_class_tokens(image_vectors)
        image_embeddings = self.image_projection(image_vectors)
        input_embeddings = torch.concat((class_embedding, image_embeddings), dim=1)
        input_embeddings = input_embeddings + self.pos[0:input_embeddings.shape[1]]
        encoded = self.encoder(input_embeddings, input_embeddings, input_embeddings)

        class_token_output = encoded[:, 0:1, ::]  # B x N x D -> B x 1 x D (class token of each sequence in batch)
        output_logits = self.classification_head(class_token_output)
        return output_logits


def img_to_patches(img_tensor, patch_size=(16, 16)):
    h, w, c = img_tensor.shape
    h_steps = list(range(0, h + 1, patch_size[0]))
    w_steps = list(range(0, w + 1, patch_size[1]))

    patches = []
    for i in range(len(h_steps) - 1):
        h_start, h_end = h_steps[i], h_steps[i+1]
        for j in range(len(w_steps) - 1):
            w_start, w_end = w_steps[i], w_steps[i+1]
            patch = img_tensor[h_start:h_end, w_start:w_end, ::].clone().detach()
            patches.append(patch)

    patches = torch.stack(patches)
    patches = torch.flatten(patches, start_dim=1)
    return patches


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float32)

    patch_size = (16, 16)
    num_channels = 3
    img_dim = patch_size[0] * patch_size[1] * num_channels

    dmodel = 1024
    hidden_dim = 2048
    heads = 4
    num_blocks = 8
    context_length = 1024   # 512x512 image will have 1024 patches
    num_classes = 100
    finetuning = False   # switch between two layer NN during training and 1 layer linear during finetuning on classification head

    # model = Transformer(context_length, dmodel, vocab_size, hidden_dim, heads)
    # out = model(torch.tensor([10, 1223, 23, 345, 234]),
    #             torch.tensor([10, 1223, 23, 345, 234, 2323, 232, 2321, 2313, 12]))
    img = torch.rand(256, 256, 3)
    patches = torch.stack([img_to_patches(img), img_to_patches(img)])
    class_label = torch.tensor([37, 37])

    model = VisionTransformer(
        context_length=context_length,
        dmodel=dmodel, num_classes=num_classes,
        hidden_dim=hidden_dim, heads=heads,
        img_dim=img_dim, num_blocks=num_blocks
    )
    output = model(patches)
    breakpoint()