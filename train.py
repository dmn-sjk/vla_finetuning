"""Phase 1: native torch.distributed training loop.

Runs identically at world size 1 (`python train.py`) and under torchrun
(`torchrun --nproc_per_node=8 train.py`). Develop on 1 GPU, scale unchanged.

The distributed structure here — sampler epoch, gradient-accumulation sync,
rank-gated I/O, count-weighted metric reduction — is invisible on one GPU and
silently wrong on eight, so it is written in from the start.
"""

import argparse
import contextlib
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src import dist_utils as D
from src.toy import ToyTrajectoryDataset, ToyVLA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32, help="per device")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-samples", type=int, default=8192)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--grad-checkpointing", action="store_true")
    # bf16 on by default (A100 supports it); --no-bf16 falls back to fp32 for
    # debugging numerics. Note `action="store_true", default=True` would make
    # the flag impossible to turn off.
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("SCRATCH", ".")) / "vla_finetuning/runs/phase1",
    )
    p.add_argument("--resume", action="store_true", help="resume from out/last.pt if present")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def build_loaders(args):
    train_ds = ToyTrajectoryDataset(args.train_samples)
    val_ds = ToyTrajectoryDataset(args.val_samples)

    train_sampler = val_sampler = None
    if D.is_distributed():
        # drop_last keeps every rank's epoch the same length, so no rank runs
        # out of batches early and deadlocks the others at the next collective.
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        # Val padding duplicates a few samples on uneven splits; the
        # count-weighted reduction below keeps the mean honest anyway.
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, sampler=train_sampler, shuffle=train_sampler is None,
        drop_last=True, **common,
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader, train_sampler, val_sampler


def save_checkpoint(path: Path, model, optimizer, epoch: int, global_step: int):
    """Rank-0 only, written atomically — a torn checkpoint on Lustre is worse
    than no checkpoint, and the 2-day wall clock means we will need to resume."""
    if not D.is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "model": raw.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"], ckpt["global_step"]


@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx):
    model.eval()
    loss_sum, n_seen = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast_ctx():
            pred = model(batch["images"], batch["state"], batch["tokens"])
            loss = nn.functional.mse_loss(pred.float(), batch["actions"].float())
        bs = batch["actions"].shape[0]
        loss_sum += loss.item() * bs
        n_seen += bs
    # Reduce sums, not means: ranks may hold different sample counts.
    total_loss = D.all_reduce_sum(loss_sum, device)
    total_n = D.all_reduce_sum(float(n_seen), device)
    model.train()
    return total_loss / max(total_n, 1.0)


def main():
    args = parse_args()
    device = D.setup()

    # Same seed on every rank so model init is identical; the sampler, not the
    # seed, is what makes ranks see different data.
    torch.manual_seed(args.seed)

    model = ToyVLA(grad_checkpointing=args.grad_checkpointing).to(device)
    if D.is_distributed():
        model = DDP(model, device_ids=[D.local_rank()] if device.type == "cuda" else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader, val_loader, train_sampler, _ = build_loaders(args)

    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_bf16 else contextlib.nullcontext
    )

    start_epoch, global_step = 0, 0
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        start_epoch, global_step = load_checkpoint(ckpt_path, model, optimizer, device)
        D.print0(f"resumed from {ckpt_path} at epoch {start_epoch}, step {global_step}")
    D.barrier()

    # The one number that makes W&B curves comparable across cluster shapes.
    effective_batch = args.batch_size * args.grad_accum * D.world_size()
    D.print0(
        f"world_size={D.world_size()} per_device_bs={args.batch_size} "
        f"grad_accum={args.grad_accum} effective_batch={effective_batch} bf16={use_bf16}"
    )

    run = None
    if args.wandb and D.is_main_process():
        import wandb
        run = wandb.init(
            project="vla-finetuning", dir=os.environ.get("WANDB_DIR"),
            config={**vars(args), "world_size": D.world_size(),
                    "effective_batch": effective_batch},
        )

    model.train()
    for epoch in range(start_epoch, args.epochs):
        # Without this, every rank reshuffles identically each epoch and the
        # sharding silently degrades. Costs one line, invisible on 1 GPU.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0, running, seen = time.time(), 0.0, 0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            is_accum = (i + 1) % args.grad_accum != 0

            # Skip the gradient all-reduce on micro-batches that are not the
            # last of their accumulation window — this is where most of the
            # speedup from gradient accumulation actually comes from.
            sync_ctx = model.no_sync if (isinstance(model, DDP) and is_accum) else contextlib.nullcontext
            with sync_ctx():
                with autocast_ctx():
                    pred = model(batch["images"], batch["state"], batch["tokens"])
                    loss = nn.functional.mse_loss(pred.float(), batch["actions"].float())
                (loss / args.grad_accum).backward()

            running += loss.item()
            seen += 1

            if not is_accum:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    # Local mean is fine for a progress line; the reduced val
                    # loss is the number to trust.
                    avg = running / max(seen, 1)
                    D.print0(f"epoch {epoch} step {global_step} loss {avg:.4f}")
                    if run is not None:
                        run.log({"train/loss": avg, "epoch": epoch}, step=global_step)
                    running, seen = 0.0, 0

        val_loss = evaluate(model, val_loader, device, autocast_ctx)
        D.print0(f"epoch {epoch} done in {time.time() - t0:.1f}s val_loss {val_loss:.4f}")
        if run is not None:
            run.log({"val/loss": val_loss, "epoch": epoch}, step=global_step)

        save_checkpoint(ckpt_path, model, optimizer, epoch + 1, global_step)
        # Hold every rank until the write lands, so a crash mid-save cannot
        # leave ranks disagreeing about what the last good checkpoint is.
        D.barrier()

    if run is not None:
        run.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
