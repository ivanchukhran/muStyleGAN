import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from util.evaluation import get_noise


class EqualizedWeight(nn.Module):
    def __init__(self, shape: tuple) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(shape))
        self.scale = 1 / np.sqrt(np.prod(shape[1:]))

    def forward(self) -> torch.Tensor:
        return self.weight * self.scale


class EqualizedLinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight((out_channels, in_channels))
        self.bias = nn.Parameter(torch.ones(out_channels) * bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)


class ModulatedConv2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            demodulate: float = True,
            eps: float = 1e-6
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.eps = eps
        self.weight = EqualizedWeight((out_channels, in_channels, kernel_size, kernel_size))
        self.padding = self.kernel_size // 2

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b, _, height, width = x.shape
        w = w[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :] * w
        if self.demodulate:
            sigma = torch.rsqrt((weights ** 2).sum([2, 3, 4]) + self.eps)
            weights = weights * sigma[:, :, None, None, None]
        x = x.reshape(1, -1, height, width)
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_channels, *ws)
        x = F.conv2d(x, weights, padding=self.padding, groups=b)
        return x.reshape(-1, self.out_channels, height, width)


class Smooth(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        kernel = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)
        x = self.pad(x)
        x = F.conv2d(x, self.kernel)
        return x.view(b, c, h, w)


class Upsample(nn.Module):
    def __init__(self, factor: int = 2, mode: str = "nearest-exact") -> None:
        super().__init__()
        match mode:
            case "nearest-exact":
                self.upsample = nn.Upsample(scale_factor=factor, mode="nearest-exact")
            case "bilinear":
                self.upsample = nn.Upsample(scale_factor=factor, mode="bilinear", align_corners=False)
            case _:
                raise ValueError(f"Unknown mode {mode}")
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        return self.smooth(x)


class Downsample(nn.Module):
    def __init__(self, factor: int = 2, mode: str = "nearest-exact") -> None:
        super().__init__()
        match mode:
            case "nearest-exact" | "bilinear":
                self.mode = mode
            case _:
                raise ValueError(f"Unknown mode {mode}")
        self.factor = factor
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.smooth(x)
        return F.interpolate(x, (x.shape[2] // self.factor, x.shape[3] // self.factor), mode=self.mode)


class EqualizedConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0):
        super().__init__()
        self.weight = EqualizedWeight((out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.ones(out_channels))
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class ToRGB(nn.Module):
    def __init__(self, w_dim: int, out_channels: int) -> None:
        super().__init__()
        self.style = EqualizedLinear(w_dim, out_channels)
        self.conv = ModulatedConv2D(out_channels, 3, kernel_size=1)
        self.bias = nn.Parameter(torch.zeros(3))
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        s = self.style(w)
        x = self.conv(x, s)
        return self.activation(x + self.bias[None, :, None, None])


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
