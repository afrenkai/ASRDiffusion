import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from models.mamba_denoiser import MambaDenoiser


def train_distributed(
    train_dl, val_dl, rank, world_size, model, loss_fn, optimizer, lr
):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    # can also compile for faster
    # ddp_model = torch.compile(ddp_model)
    loss_fn = loss_fn()
    optim = optimizer(ddp_model.parameters(), lr=lr)

    out = ddp_model(train_dl.to(rank))
    labs = val_dl.to(rank)
    loss_fn(out, labs)
    optimizer.step()


def main():
    world_size = 2
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size, join=True)
