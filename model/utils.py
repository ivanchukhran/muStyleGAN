from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from scipy.stats import truncnorm
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

import json


def read_settings(path: str) -> dict:
    """
    Read the settings from the given path.
    :param path: The path to the settings file.
    :return: settings: dict - The settings.
    """
    try:
        with open(path, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"Error reading settings file: {e}")
        settings = {}
    return settings


def show_tensor_images(image_tensor, num_images=16, save_path: Optional[str] = None):
    """
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflatten = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflatten[:num_images], padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()


def get_noise(shape: tuple, device='cuda') -> torch.Tensor:
    """
    Generate a noise tensor with the given channels.
    :param shape: The shape of the noise tensor.
    :param device: The device to run the noise tensor on.
    :return: noise: Tensor - The generated noise.
    """
    return torch.randn(shape, device=device)


def truncated_noise(n_samples: int, z_dim: int, truncation: float) -> torch.Tensor:
    """
    Generate a truncated noise tensor with the given channels.
    :param n_samples: The number of samples to generate.
    :param z_dim: The dimensionality of the noise vector.
    :param truncation: The truncation factor to use.
    :return: noise: Tensor - The generated noise.
    """
    values = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(values)


def gradient_of(critic, real_images: torch.Tensor, fake_images: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
    """
    Get the gradient of the critic's scores with respect to mixes of real and fake images.
    :param critic: The critic model.
    :param real_images: Tensor - The real images.
    :param fake_images: Tensor - The fake images.
    :param lambda_: The interpolation factor.
    :return: gradient: Tensor - The gradient of the critic's scores with respect to mixes of real and fake images.
    """
    mixed_images = real_images * lambda_ + fake_images * (1 - lambda_)
    mixed_scores = critic(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True
    )[0]
    return gradient


def gradient_penalty(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    """
    batch_size = x.shape[0]
    gradients, *_ = torch.autograd.grad(outputs=f,
                                        inputs=x,
                                        grad_outputs=f.new_ones(f.shape),
                                        create_graph=True,
                                        allow_unused=True)
    gradients = gradients.reshape(batch_size, -1)
    norm = gradients.norm(2, dim=-1)
    return torch.mean((norm - 1) ** 2)


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Calculate the generator loss given fake scores.
    :param fake_scores: Tensor - The fake scores.
    :return: loss: Tensor - The generator loss.
    """
    return -torch.mean(fake_scores)


def discriminator_loss(real_scores: torch.Tensor,
                       fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Calculate the critic loss given real and fake scores, and the gradient penalty.
    :param real_scores: Tensor - The real scores.
    :param fake_scores: Tensor - The fake scores.
    :return: loss: Tensor - The critic loss.
    """
    return F.relu((1 - real_scores).mean()) + F.relu((1 + fake_scores).mean())
