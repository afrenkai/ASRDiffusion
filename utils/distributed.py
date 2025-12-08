import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()