import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def calc_ouput_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def show_tensor_images(image_tensor):
    # Detach tensor and ensure it's on CPU, then take the first image if batch exists
    img = image_tensor.detach().cpu()

    # If the image has a single channel, treat it as grayscale
    if img.dim() == 2 or img.shape[0] == 1:
        plt.imshow(img.squeeze(), cmap='gray')
    else:
        # If image is normalized (using the usual ImageNet means and stds), de-normalize it
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1)
        # Convert tensor (C, H, W) → (H, W, C) for RGB plotting
        plt.imshow(img.permute(1, 2, 0))

    plt.axis("off")
    plt.show()
