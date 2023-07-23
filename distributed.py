import torch


def setup_device(rank: int = 0) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(rank)
    return device
