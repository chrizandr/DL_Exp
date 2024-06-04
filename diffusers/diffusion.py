import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms


beta_start = 1e-4
beta_end = 0.02
num_steps = 1000
beta_range = torch.arange(beta_start, beta_end, step=(beta_end - beta_start) / num_steps)
alpha_range = 1 - beta_range
alpha_prods = torch.cumprod(alpha_range, dim=0)


def forward_diffusion_one_step(img, timestep):
    alpha_t = alpha_prods[timestep]
    b = torch.sqrt(1-alpha_t)
    a = torch.sqrt(alpha_t)
    epsilon = torch.normal(torch.zeros_like(img), torch.eye(img.shape[0], img.shape[1]))
    x_t = a * img + b * epsilon
    # breakpoint()
    return x_t


def forward_diffusion_iterative(img, timesteps):
    x_t = torch.clone(img)
    for i in range(timesteps):
        beta = beta_range[i]
        mean = torch.sqrt(1-beta) * x_t
        std = beta * torch.eye(img.shape[0], img.shape[1])
        x_t = torch.normal(mean, std)

    return x_t

if __name__ == "__main__":
    image = Image.open("lena.png").convert('RGB')
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts the image to a PyTorch tensor (H, W, C) to (C, H, W) and scales it to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet mean and std
    ])
    normalized_tensor = transform(image)
    breakpoint()
    out = forward_diffusion_one_step(img, 1)
    breakpoint()