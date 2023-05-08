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
    return F.interpolate(smaller_image, size=larger_image.shape[-2:], mode=interp_mode)


def get_gradient(critic, real_images: torch.Tensor, fake_images: torch.Tensor, lambda_: float) -> torch.Tensor:
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
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient) -> torch.Tensor:
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty


def generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Calculate the generator loss given fake scores.
    :param fake_scores: Tensor - The fake scores.
    :return: loss: Tensor - The generator loss.
    """
    return -torch.mean(fake_scores)


def critic_loss(real_scores: torch.Tensor,
                fake_scores: torch.Tensor,
                gradient_penalty: torch.Tensor,
                penalty_weight: float) -> torch.Tensor:
    """
    Calculate the critic loss given real and fake scores, and the gradient penalty.
    :param real_scores: Tensor - The real scores.
    :param fake_scores: Tensor - The fake scores.
    :param gradient_penalty: Tensor - The gradient penalty.
    :param penalty_weight: float - The weight to apply to the gradient penalty.
    :return: loss: Tensor - The critic loss.
    """
    return torch.mean(fake_scores) - torch.mean(real_scores) + gradient_penalty * penalty_weight
