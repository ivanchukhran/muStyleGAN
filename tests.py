from model.components import *
from utils import truncated_noise


def test_truncation():
    assert tuple(truncated_noise(n_samples=10, z_dim=5, truncation=0.7).shape) == (10, 5)
    simple_noise = truncated_noise(n_samples=1000, z_dim=10, truncation=0.2)
    assert 0.199 < simple_noise.max() < 2
    assert -0.199 > simple_noise.min() > -0.2
    assert 0.113 < simple_noise.std() < 0.117
    print("Success!")


def test_mapping_layers():
    map_fn = MappingLayers(10, 20, 30)
    outputs = map_fn(torch.randn(1000, 10))
    assert tuple(map_fn(torch.randn(2, 10)).shape) == (2, 30)
    assert len(map_fn.mapping) > 4
    assert 0.05 < outputs.std() < 0.3
    assert -2 < outputs.min() < 0
    assert 2 > outputs.max() > 0
    layers = [str(x).replace(' ', '').replace('inplace=True', '') for x in map_fn.mapping]
    assert layers == ['Linear(in_features=10,out_features=20,bias=True)',
                      'ReLU()',
                      'Linear(in_features=20,out_features=20,bias=True)',
                      'ReLU()',
                      'Linear(in_features=20,out_features=30,bias=True)']
    print("Success!")


def test_noise_injection():
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


def test_adain():
    pass


if __name__ == '__main__':
    test_truncation()
    test_mapping_layers()
    test_noise_injection()
    test_adain()
    print('All tests passed!')
