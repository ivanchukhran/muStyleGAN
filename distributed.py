import os
from typing import List

import torch
from torch import nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from local_secrets import MASTER_ADDR


def setup_ddp(rank: int, world_size: int, backend: str = "gloo") -> None:
    # TODO: change localhost to the masters address
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def destroy_ddp() -> None:
    destroy_process_group()


def wrap_to_ddp(to_wrap: nn.Module | List[nn.Module], device_ids: int | List = 0) -> nn.Module | List[nn.Module]:
    match to_wrap:
        case nn.Module:
            return DDP(to_wrap, device_ids=[device_ids])
        case [*to_wrap]:
            return [DDP(module) for module in to_wrap]
        case _:
            raise TypeError(f"Expected nn.Module or list of nn.Module, got {type(to_wrap)}")
