import torch
from torch import nn
from torch.nn import functional as F

from scipy.stats import truncnorm


def get_noise(shape: tuple, device='cuda') -> torch.Tensor:
    """
    Generate a noise tensor with the given channels.
    :param shape: The shape of the noise tensor.
    :param device: The device to run the noise tensor on.
    :return: noise: Tensor - The generated noise.
    """
    return torch.randn(shape, device=device)


def get_truncated_noise(n_samples: int, z_dim: int, truncation: float) -> torch.Tensor:
    """
    Generate a truncated noise tensor with the given channels.
    :param n_samples: The number of samples to generate.
    :param z_dim: The dimensionality of the noise vector.
    :param truncation: The truncation factor to use.
    :return: noise: Tensor - The generated noise.
    """
    values = truncnorm.rvs(-truncation, truncation, size=(n_samples, z_dim))
    return torch.Tensor(values)


def upsample(smaller_image: torch.Tensor, larger_image: torch.Tensor, interp_mode='bilinear'):
    """
    Upsample the smaller image to match the size of the larger image.
    :param smaller_image: Tensor - The smaller image.
    :param larger_image: Tensor - The larger image.
    :param interp_mode: The interpolation mode to use for upsampling.
    :return: upsampled_image: Tensor - The upsampled image.
    """
    return F.interpolate(smaller_image, size=larger_image.shape[2:], mode=interp_mode)