"""Distributed helpers for Phase 1.

Everything here is a no-op when the script is launched without torchrun, so the
same train.py runs on a laptop, on one A100, or on 8 via torchrun.
"""

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
    """Gate for W&B, checkpoint writes, dataset downloads, progress bars.

    Concurrent writes from 8 ranks to shared Lustre corrupt files, so anything
    touching disk or an external service goes behind this.
    """
    return rank() == 0


def setup() -> torch.device:
    """Bind this process to its GPU and join the process group."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank())
        device = torch.device("cuda", local_rank())
    else:
        device = torch.device("cpu")

    if is_distributed():
        # Long timeout: the first NCCL collective on a cold cluster can be slow,
        # and a too-short timeout looks identical to a real hang.
        # device_id pins collectives to this rank's GPU; without it barrier()
        # infers a device from ambient context and can pick the wrong one.
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(minutes=30),
            device_id=device if device.type == "cuda" else None,
        )
    return device


def cleanup() -> None:
    """Barrier then tear down. Skipping this leaves jobs hanging at exit."""
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
