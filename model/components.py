from typing import Union, Optional

import torch
from scipy.stats import truncnorm
from torch import nn
from torch.nn import functional as F

from utils import get_noise, upsample


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, image) -> torch.Tensor:
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        return image + self.weight * get_noise(noise_shape, device=image.device)

    def get_self(self) -> "NoiseInjection":
        return self

    @property
    def get_weight(self) -> nn.Parameter:
        return self.weight


class MappingLayers(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, w_dim: int):
        super().__init__()
        self.map = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, w_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Takes a noise vector and returns a latent vector representation."""
        return self.map(z)

    @property
    def mapping(self) -> nn.Sequential:
        return self.map


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels)
        self.scale = nn.Linear(w_dim, channels)
        self.shift = nn.Linear(w_dim, channels)

    def forward(self, image, w) -> torch.Tensor:
        normalized_image = self.instance_norm(image)
        style_scale = self.scale(w)[:, :, None, None]
        style_shift = self.shift(w)[:, :, None, None]
        return style_scale * normalized_image + style_shift

    @property
    def scale_transform(self) -> nn.Linear:
        return self.scale

    @property
    def shift_transform(self) -> nn.Linear:
        return self.shift

    @property
    def get_self(self) -> "AdaIN":
        return self


class GeneratorBlock(nn.Module):
    """
    A microgenerator block.
    :param in_channels: The number of channels in the input.
    :param out_channels: The number of channels in the output.
    :param w_dim: The dimension of the latent vector.
    :param kernel_size: The size of the kernel.
    :param starting_size: The starting size of the image.
    :param relu_slope: The slope of the leaky ReLU.
    :param upsample_mode: The mode to use for upsampling. By default, bilinear. If None, no upsampling is performed.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            w_dim: Union[int, tuple],
            kernel_size,
            starting_size,
            relu_slope: float = 0.2,
            upsample_mode: Optional[str] = 'bilinear'
    ):
        super().__init__()
        self.upsample_mode = upsample_mode
        if self.upsample_mode:
            self.upsample = nn.Upsample((starting_size, starting_size), mode=self.upsample_mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.inject_noise = NoiseInjection(out_channels)
        self.adain = AdaIN(out_channels, w_dim)
        self.activation = nn.LeakyReLU(negative_slope=relu_slope)

    def forward(self, x, w) -> torch.Tensor:
        if self.upsample_mode:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        return self.activation(x)


class ModulatedConv2D(nn.Module):
    def __init__(
            self,
            w_dim: Union[int, tuple],
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            padding: int = 1,
            eps: float = 1e-6
    ):
        super().__init__()
        self.conv_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.style_scale_transform = nn.Linear(w_dim, out_channels)
        self.eps = eps
        self.padding = padding

    def forward(self, x, w) -> torch.Tensor:
        style_scale = self.style_scale_transform(w)
        w_prime = self.conv_weight[None] * style_scale[:, None, :, None, None]
        w_prime_prime = w_prime / torch.sqrt(
            (w_prime ** 2).sum([2, 3, 4])[:, :, None, None, None] + self.eps
        )
        batch_size, in_channels, height, width = x.shape
        out_channels = w_prime_prime.shape[2]
        efficient_x = x.view(1, batch_size * in_channels, height, width)
        efficient_filter = w_prime_prime.view(batch_size * out_channels, in_channels, *w_prime_prime.shape[3:])
        efficient_out = F.conv2d(efficient_x, efficient_filter, padding=self.padding, groups=batch_size)
        return efficient_out.view(batch_size, out_channels, *x.shape[2:])


class Generator(nn.Module):
    def __init__(
            self,
            z_dim,
            map_hidden_dim,
            w_dim,
            in_chan,
            out_chan,
            kernel_size,
            hidden_chan,
            interp_alpha: float = 0.2
    ):
        super().__init__()
        self.mapping = MappingLayers(z_dim=z_dim, hidden_dim=map_hidden_dim, w_dim=w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, hidden_chan, 4, 4))
        self.block0 = GeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, upsample_mode=None)
        self.block1 = GeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = GeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)

        # (Note that this is simplified, with clipping used in the real StyleGAN)
        self.block1_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.block2_to_image = nn.Conv2d(hidden_chan, out_chan, kernel_size=1)
        self.alpha = interp_alpha

    def forward(self, z: torch.Tensor, return_intermediate: bool = False) -> Union[torch.Tensor, tuple]:
        w = self.mapping(z)
        x = self.starting_constant
        x = self.block0(x, w)
        x_small = self.block1(x, w)
        x_small_image = self.block1_to_image(x_small)
        x_big = self.block2(x_small, w)
        x_big_image = self.block2_to_image(x_big)
        x_small_upsample = upsample(x_small_image, x_big_image)
        interpolation = torch.lerp(x_small_upsample, x_big_image, self.alpha)
        return interpolation, x_small_upsample, x_big_image if return_intermediate else interpolation


class CriticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, relu_slope):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(negative_slope=relu_slope)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class CriticEpilogue(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class Critic(nn.Module):
    def __init__(
            self, # TODO: Fil in
    ):
        super().__init__()
        self.blocks = nn.Sequential() # TODO: Fill in

    def forward(self, x):
        pass


# decide whether to use this method or not.
    def make_block(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, tuple],
            stride: int,
            padding: int,
            relu_slope,
            is_last: bool = False
    ):
        return (CriticEpilogue(in_channels, out_channels, kernel_size, stride, padding) if is_last else
                CriticBlock(in_channels, out_channels, kernel_size, stride, padding, relu_slope))
