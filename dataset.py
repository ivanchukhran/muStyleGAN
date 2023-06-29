import os
from typing import Optional, Tuple

import torch.utils.data
import torchvision.transforms
from PIL import Image


def default_transformation(crop_size: int | Tuple[int, int],
                           transformation: Optional[torchvision.transforms.Compose] = None):
    if not transformation:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop((crop_size, crop_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
    return transformation


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, crop_size: int = 32, transform: Optional[torchvision.transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = default_transformation(crop_size, transformation=transform)

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
