from model.networks import *
from model.utils import truncated_noise


def test_truncation():
    print(f"{'-' * 10}Testing truncated noise...{'-' * 10}")
    assert tuple(truncated_noise(n_samples=10, z_dim=5, truncation=0.7).shape) == (10, 5)
    simple_noise = truncated_noise(n_samples=1000, z_dim=10, truncation=0.2)
    assert 0.199 < simple_noise.max() < 2
    assert -0.199 > simple_noise.min() > -0.2
    assert 0.113 < simple_noise.std() < 0.117
    print("Success!")


def test_mapping_layers():
    print(f"{'-' * 10}Testing mapping layers...{'-' * 10}")
    map_fn = MappingNetwork(z_dim=10, w_dim=20, n_layers=2)
    outputs = map_fn(torch.randn(1000, 10))
    assert tuple(outputs.shape) == (1000, 20), f"Bad output shape: {outputs.shape}, expected: {(1000, 20)}"
    assert str(map_fn) == """Linear(10, 20)
LeakyReLU(0.2)
Linear(20, 20)
LeakyReLU(0.2)""", f"Bad string representation: {str(map_fn)}"
    print("Success!")


def test_noise_injection():
    print(f"{'-' * 10}Testing noise injection...{'-' * 10}")
    test_noise_channels = 3000
    test_noise_samples = 20
    fake_images = torch.randn(test_noise_samples, test_noise_channels, 10, 10)
    inject_noise = NoiseInjection(test_noise_channels)
    assert torch.abs(inject_noise.weight.std() - 1) < 0.1
    assert torch.abs(inject_noise.weight.mean()) < 0.1
    assert type(inject_noise.weight) == torch.nn.parameter.Parameter

    assert tuple(inject_noise.weight.shape) == (1, test_noise_channels, 1, 1)
    inject_noise.weight = nn.Parameter(torch.ones_like(inject_noise.weight))
    # Check that something changed
    assert torch.abs((inject_noise(fake_images) - fake_images)).mean() > 0.1
    # Check that the change is per-channel
    assert torch.abs((inject_noise(fake_images) - fake_images).std(0)).mean() > 1e-4
    assert torch.abs((inject_noise(fake_images) - fake_images).std(1)).mean() < 1e-4
    assert torch.abs((inject_noise(fake_images) - fake_images).std(2)).mean() > 1e-4
    assert torch.abs((inject_noise(fake_images) - fake_images).std(3)).mean() > 1e-4
    # Check that the per-channel change is roughly normal
    per_channel_change = (inject_noise(fake_images) - fake_images).mean(1).std()
    assert 0.9 < per_channel_change < 1.1
    # Make sure that the weights are being used at all
    inject_noise.weight = nn.Parameter(torch.zeros_like(inject_noise.weight))
    assert torch.abs((inject_noise(fake_images) - fake_images)).mean() < 1e-4
    assert len(inject_noise.weight.shape) == 4
    print("Success!")


def test_modulated_convolution():
    print(f"{'-' * 10} Testing modulated convolution {'-' * 10}")
    batch_size = 10
    in_channels = 256
    out_channels = 128
    kernel_size = 3
    resolution = 4
    w_dim = 256

    noise = torch.randn((batch_size, in_channels, resolution, resolution))
    w = torch.randn((batch_size, w_dim))
    modulated_conv = ModulatedConv2D(in_channels, out_channels, kernel_size)
    out = modulated_conv(noise, w)
    print(f"input: {noise.shape}, {w.shape} => out: {out.shape}")
    assert tuple(out.shape) == (batch_size, out_channels, resolution,
                                resolution), f"Invalid output shape, expected: " \
                                             f"{(batch_size, out_channels, resolution, resolution)}, " \
                                             f"got: {out.shape}"
    print("Success!")


def test_synthesis_layer() -> None:
    batch_size = 10
    hidden_dim = 256
    z_dim = 128
    w_dim = 128
    in_channels = 256
    out_channels = 128
    resolution = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Testing synthesis layer...")
    z = torch.randn((batch_size, z_dim))
    mapping = MappingNetwork(z_dim, hidden_dim, w_dim)
    noise = get_noise((batch_size, z_dim, 4, 4), device='cpu')
    print(f"noise: {z.shape}")
    w = mapping(z)
    print(f"w: {w.shape}")
    print(f"in_channels: {in_channels}, out_channels: {out_channels}, w_dim: {w_dim}, resolution: {resolution}")
    layer = SynthesisLayer(in_channels, out_channels, w_dim, resolution)
    out = layer(noise, w)
    print(f"out: {out.shape}")
    assert tuple(out.shape) == (batch_size, out_channels, 4, 4)

    print("Success!")


def test_synthesis_block() -> None:
    print(f"{'-' * 10} Testing synthesis block {'-' * 10}")
    batch_size = 10
    hidden_dim = 256
    z_dim = 128
    w_dim = 128


if __name__ == '__main__':
    test_truncation()
    test_mapping_layers()
    test_modulated_convolution()
    test_noise_injection()
    # test_synthesis_layer()
    print('All tests passed!')
