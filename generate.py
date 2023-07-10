import PIL.Image
import torch
from torchvision.transforms import transforms

from util.files import fetch_json
from util.evaluation import get_noise
from model.generator import Generator

from constants import SETTINGS_PATH

SETTINGS = fetch_json(SETTINGS_PATH)


def to_image(tensor: torch.Tensor) -> PIL.Image.Image:
    return transforms.ToPILImage()(tensor.cpu())


def generate(num_images: int, generator_type: str = "stylegan2-landscape-32") -> PIL.Image.Image | list[PIL.Image.Image]:
    settings = SETTINGS.get(generator_type)
    if settings is None:
        raise ValueError(f"Generator {generator_type} not supported.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(**settings).from_pretrained(settings.get("weights_root")).to(device)
    z = get_noise((num_images, settings.get("z_dim")), device=device)
    print(f"generated nosie: {z.shape}")
    generated_tensor = generator(z)
    if num_images == 1:
        return to_image(torch.squeeze(generated_tensor))
    return [to_image(image) for image in generated_tensor]
