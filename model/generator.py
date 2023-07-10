from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from model.components import EqualizedLinear, ModulatedConv2D, NoiseInjection, ToRGB, Upsample


class Generator(nn.Module):
    def __init__(self,
                 z_dim: int,
                 w_dim: int,
                 n_mapping_layers: int,
                 image_resolution: int,
                 *args, **kwargs) -> None:
        super().__init__()
        self.mapping_network = MappingNetwork(z_dim, w_dim, n_mapping_layers)
        self.synthesis_network = SynthesisNetwork(image_resolution, w_dim)

    @property
    def mapping(self) -> "MappingNetwork":
        return self.mapping_network

    @property
    def synthesis(self) -> "SynthesisNetwork":
        return self.synthesis_network

    def from_pretrained(self, path: str) -> "Generator":
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
        return self

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self.mapping_network(z)
        return self.synthesis_network(w)


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


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int, n_layers: int, relu: str = "leaky", slope: float = 0.2) -> None:
        super().__init__()
        match relu:
            case "leaky":
                activation = nn.LeakyReLU(negative_slope=slope)
            case "relu":
                activation = nn.ReLU()
            case _:
                print(f"Activation {relu} not supported. Using LeakyReLU instead.")
                activation = nn.LeakyReLU(negative_slope=slope)
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(z_dim, w_dim))
            else:
                layers.append(nn.Linear(w_dim, w_dim))
            layers.append(activation)
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
                case nn.ReLU():
                    names.append(f"ReLU({module.negative_slope})")
                case nn.LeakyReLU():
                    names.append(f"LeakyReLU({module.negative_slope})")
        return '\n'.join(names)
