import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm


class GradientPenalty(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(
            outputs=d, inputs=x,
            grad_outputs=d.new_ones(d.shape),
            create_graph=True
        )
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=1)
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        image_size = x.shape[2] * x.shape[3]
        y = torch.randn(x.shape, device=device)
        output = (x * y).sum() / np.sqrt(image_size)
        gradients, *_ = torch.autograd.grad(outputs=output, inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()
        if self.steps > 0:
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            loss = torch.mean((norm - a) ** 2)
        else:
            loss = norm.new_tensor(0)
        mean = norm.mean().detach()
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        self.steps.add_(1.)
        return loss


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
        :param x: gradient input tensor
        :param f: gradient output tensor
    Returns:
        penalty: the gradient penalty of the given tensors
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