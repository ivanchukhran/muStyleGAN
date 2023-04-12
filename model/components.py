from typing import Union, Optional

import torch
from torch import nn
from torch.nn import functional as F

from utils import get_noise, upsample


class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image):
        noise_shape = (image.shape[0], 1, image.shape[2], image.shape[3])
        return image + self.weight * get_noise(noise_shape, device=image.device)

    @property
    def weight(self) -> nn.Parameter:
        return self.weight

    def get_self(self) -> "NoiseInjection":
        return self


class muMappingLayers(nn.Module):
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

    def forward(self, image, w):
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


class MuGeneratorBlock(nn.Module):
    """
    A micro generator block.
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

    def forward(self, x, w):
        if self.upsample_mode:
            x = self.upsample(x)
        x = self.conv(x)
        x = self.inject_noise(x)
        x = self.adain(x, w)
        return self.activation(x)


class muGenerator(nn.Module):
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
        self.mapping = muMappingLayers(z_dim=z_dim, hidden_dim=map_hidden_dim, w_dim=w_dim)
        self.starting_constant = nn.Parameter(torch.randn(1, hidden_chan, 4, 4))
        self.block0 = MuGeneratorBlock(in_chan, hidden_chan, w_dim, kernel_size, 4, upsample_mode=None)
        self.block1 = MuGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 8)
        self.block2 = MuGeneratorBlock(hidden_chan, hidden_chan, w_dim, kernel_size, 16)

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
