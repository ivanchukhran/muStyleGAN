import numpy as np
import torch
from torch import nn

from model.components import EqualizedLinear, EqualizedConv2D, Downsample


class AdaIN(nn.Module):
    def __init__(self, channels, w_dim) -> None:
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


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Downsample(),
            EqualizedConv2D(in_channels, out_channels, kernel_size=1)
        )
        self.block = nn.Sequential(
            EqualizedConv2D(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            EqualizedConv2D(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.downsample = Downsample()
        self.scale = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.block(x)
        x = self.downsample(x)
        x = (x + residual) * self.scale
        return x


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4, eps: float = 1e-8) -> None:
        super().__init__()
        self.group_size = group_size
        self.epsilon = eps

    def forward(self, x) -> torch.Tensor:
        grouped = x.reshape(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + self.epsilon)
        std = std.mean().view(1, 1, 1, 1)
        b, c, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, resolution: int, n_features: int = 64, max_features: int = 512) -> None:
        super().__init__()
        self.from_rgb = nn.Sequential(
            EqualizedConv2D(3, n_features, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.2)
        )
        log_resolution = int(np.log2(resolution))
        features = [min(n_features * 2 ** i, max_features) for i in range(log_resolution - 1)]
        self.n_blocks = len(features) - 1

        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(self.n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.std_dev = MiniBatchStdDev()
        final_features = features[-1] + 1
        self.conv = EqualizedConv2D(final_features, final_features, kernel_size=3)
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)
        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.final(x)
        return x
