"""Distributed helpers; all are no-ops when launched without torchrun."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed() -> bool:
    return world_size() > 1


def is_main_process() -> bool:
    """Gate for W&B, checkpoint writes, downloads and progress output."""
    return rank() == 0


def setup() -> torch.device:
    """Bind this process to its GPU and join the process group."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        device = torch.device("cuda", local_rank())
    else:
        device = torch.device("cpu")

    if is_distributed():
        # device_id pins collectives to this GPU
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=30),
            device_id=device if device.type == "cuda" else None,
        )
    return device


def cleanup() -> None:
    """Barrier then tear down, so jobs do not hang at exit."""
    if is_distributed() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed() and dist.is_initialized():
        dist.barrier()


def all_reduce_sum(value: float, device: torch.device) -> float:
    """Sum a scalar across ranks. Returns the value unchanged on 1 GPU."""
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def print0(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs, flush=True)
