from typing import Optional

import torch
from torchvision.utils import make_grid


def show_tensor_images(image_tensor: torch.Tensor, num_images=16, save_path: Optional[str] = None):
    """
    Function for visualizing images: Given a tensor of images, number of images,
    size per image, and images per row, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflatten = image_tensor.detach().cpu().clamp_(0, 1)
    image_grid = make_grid(image_unflatten[:num_images], padding=0)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()
    if save_path:
        plt.savefig(save_path)
    plt.close()
