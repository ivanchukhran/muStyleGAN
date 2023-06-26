import os
from typing import Optional, Tuple

import torch.utils.data
import torchvision.transforms
from PIL import Image


def exists(path):
    return os.path.exists(path)


def create_dir(path):
    if not exists(path):
        os.makedirs(path)
    else:
        print(f"Path {path} already exists!")


def similar_dir_exists(path: str, dir_name: str):
    for item in os.listdir(path):
        if dir_name in item:
            return True
    return False


def list_dir(path):
    return os.listdir(path)


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

        self.images = list_dir(root_dir)

    def __getitem__(self, item):
        with Image.open(os.path.join(self.root_dir, self.images[item])) as img:
            if self.transform:
                img = self.transform(img)
        return img

    def __len__(self):
        return len(self.images)
