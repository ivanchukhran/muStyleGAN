from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils import get_noise


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


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_layers: int, relu: str = "leaky", slope: float = 0.2) -> None:
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(z_dim, w_dim))
            else:
                layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(negative_slope=slope))
        self.map = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Takes a noise vector and returns a latent vector representation."""
        return self.map(z)

    @property
    def mapping(self) -> nn.Sequential:
        return self.map

    def __repr__(self):
        names = []
        for name, module in self.map.named_children():
            match module:
                case nn.Linear():
                    names.append(f"Linear({module.in_features}, {module.out_features})")
                case nn.LeakyReLU():
                    names.append(f"LeakyReLU({module.negative_slope})")
        return '\n'.join(names)


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


class Generator(nn.Module):
    def __init__(self,
                 z_dim: int,
                 w_dim: int,
                 n_mapping_layers: int,
                 image_resolution: int,
                 ) -> None:
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim, w_dim, n_mapping_layers)
        self.synthesis_network = SynthesisNetwork(image_resolution, w_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping_network(z)
        return self.synthesis_network(w)


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


class SynthesisLayer(nn.Module):
    def __init__(self, w_dim: int, in_channels: int, out_channels: int, activation: str = "leaky") -> None:
        super().__init__()
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.to_style = EqualizedLinear(w_dim, in_channels)
        self.conv = ModulatedConv2D(in_channels, out_channels)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise_injection = NoiseInjection(out_channels)
        match activation:
            case "leaky":
                self.activation = nn.LeakyReLU(negative_slope=0.2)
            case "relu":
                self.activation = nn.ReLU()
            case _:
                print(f"Activation {activation} not supported. Using LeakyReLU instead.")
                self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: bool = True) -> torch.Tensor:
        s = self.to_style(w)
        x = self.conv(x, s)
        if noise:
            x = self.noise_injection(x)
        return self.activation(x + self.bias[None, :, None, None])


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


class SynthesisBlock(nn.Module):
    def __init__(self, w_dim: int, in_channels: int, out_channels: int, activation: str = "leaky") -> None:
        super().__init__()

        self.layer1 = SynthesisLayer(w_dim, in_channels, out_channels, activation)
        self.layer2 = SynthesisLayer(w_dim, out_channels, out_channels, activation)
        self.to_rgb = ToRGB(w_dim, out_channels)

    def forward(self,
                x: torch.Tensor,
                w: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.layer1(x, w)
        x = self.layer2(x, w)
        return x, self.to_rgb(x, w)


class SynthesisNetwork(nn.Module):
    def __init__(self, resolution: int, w_dim: int, n_features: int = 32, max_features: int = 512):
        super().__init__()
        log_resolution = int(np.log2(resolution))
        features = [min(n_features * 2 ** i, max_features) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.initial_constant = nn.Parameter(torch.randn(1, features[0], 4, 4))
        self.style_block = SynthesisLayer(w_dim, features[0], features[0])
        self.to_rgb = ToRGB(w_dim, features[0])

        self.blocks = nn.ModuleList(
            [SynthesisBlock(w_dim, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        )

        self.upsample = Upsample()

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        batch_size = w.shape[0]
        x = self.initial_constant.expand(batch_size, -1, -1, -1)
        x = self.style_block(x, w)
        rgb = self.to_rgb(x, w)
        for i in range(1, self.n_blocks):
            x = self.upsample(x)
            x, new_rgb = self.blocks[i - 1](x, w)
            rgb = self.upsample(rgb) + new_rgb
        return rgb


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
        if x.shape[0] % self.group_size != 0:
            raise ValueError(f"Batch size must be a multiple of {self.group_size}")
        grouped = x.reshape(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim=0) + self.epsilon)
        std = std.mean().view(1, 1, 1, 1)
        b, c, h, w = x.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([x, std], dim=1)


class Discriminator(nn.Module):
    def __init__(self, resolution: int, n_features: int = 32, max_features: int = 512) -> None:
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


class GradientPenalty(nn.Module):
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
