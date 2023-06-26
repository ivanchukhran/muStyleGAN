import os
from typing import Optional, Tuple

import torch.utils.data
import torchvision.transforms
from PIL import Image


def create_dir_or_ignore(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print(f"Directory {path} already exists.")


def filter_by_dirname(path, dir_name) -> list[str]:
    return [folder for folder in os.listdir(path) if dir_name in folder]


def default_transformation(crop_size: int | Tuple[int, int] = 32,
                           transformation: Optional[torchvision.transforms.Compose] = None):
    if not transformation:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((crop_size, crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    return transformation


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform: Optional[torchvision.transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = default_transformation(transformation=transform)

        if os.path.exists(root_dir):
            self.images = os.listdir(root_dir)
        else:
            raise FileNotFoundError(f"Directory {root_dir} does not exist.")

    def __getitem__(self, item):
        with Image.open(os.path.join(self.root_dir, self.images[item])) as img:
            if self.transform:
                img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)
