"""LoRA fine-tuning of pi0 on LIBERO with native DDP. Runs as plain python or under torchrun."""

import argparse
import contextlib
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from torch.nn.parallel import DistributedDataParallel as DDP

from src import dist_utils as D
from src.cli import add_common_args
from src.data import build_datasets, build_loaders
from src.paths import run_dir
from src.pi0 import atomic_torch_save, autocast_ctx, evaluate, make_processors, wrap_lora
from src.timing import ThroughputTimer


def parse_args():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--lr", type=float, default=1e-4, help="default suits LoRA; full FT uses 2.5e-5")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=run_dir("pi0_libero"))
    p.add_argument(
        "--no-find-unused-parameters",
        dest="find_unused_parameters",
        action="store_false",
        default=True,
        help="disable find_unused_parameters; only safe with --lora-target default",
    )
    return p.parse_args()


def trainable_state_dict(policy) -> dict:
    """Only the LoRA parameters; the frozen base is on the Hub."""
    return {k: v.detach().cpu() for k, v in policy.state_dict().items() if "lora" in k.lower()}


def _export_adapter(raw, dest: Path):
    """Write a PEFT adapter dir atomically (build beside, then rename)."""
    tmp = dest.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    raw.save_pretrained(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)


def save_checkpoint(
    out_dir: Path,
    path: Path,
    policy,
    optimizer,
    epoch: int,
    global_step: int,
    best_val: float,
    is_best: bool,
):
    """Rank-0 only, atomic. `last.pt` resumes the run and `adapter/` is its export;
    `best.pt`/`adapter_best` hold the best val epoch.
    """
    if not D.is_main_process():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = policy.module if isinstance(policy, DDP) else policy
    payload = {
        "lora": trainable_state_dict(raw),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
    }
    atomic_torch_save(payload, path)

    _export_adapter(raw, out_dir / "adapter")
    if is_best:
        atomic_torch_save(payload, out_dir / "best.pt")
        _export_adapter(raw, out_dir / "adapter_best")


def load_checkpoint(path: Path, policy, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = policy.module if isinstance(policy, DDP) else policy
    _, unexpected = raw.load_state_dict(ckpt["lora"], strict=False)
    if unexpected:
        raise RuntimeError(
            f"checkpoint has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}"
        )
    optimizer.load_state_dict(ckpt["optimizer"])
    # keep best_val across resume so a worse epoch cannot overwrite adapter_best
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_val", float("inf"))


def main():
    args = parse_args()
    device = D.setup()

    # same seed on every rank so LoRA init matches; the sampler splits the data
    torch.manual_seed(args.seed)

    meta = LeRobotDatasetMetadata(args.dataset)
    cfg = PI0Config(
        pretrained_path=args.policy,
        device=device.type,
        dtype="bfloat16" if args.bf16 else "float32",
        gradient_checkpointing=args.grad_checkpointing,
    )

    train_ds, val_ds, n_train_eps, n_val_eps = build_datasets(args, meta, cfg)
    train_loader, val_loader, train_sampler = build_loaders(args, train_ds, val_ds)

    policy = make_policy(cfg, ds_meta=meta)
    policy = wrap_lora(policy, args.lora_target, args.lora_r, args.lora_alpha, args.lora_dropout)
    policy.to(device)

    preprocessor, _ = make_processors(cfg, meta, device, args.policy)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in policy.parameters())
    if n_trainable < 1e6:
        raise RuntimeError(
            f"only {n_trainable} trainable params for --lora-target={args.lora_target}; "
            "the target_modules regex probably matched nothing"
        )

    if D.is_distributed():
        policy = DDP(
            policy,
            device_ids=[D.local_rank()] if device.type == "cuda" else None,
            find_unused_parameters=args.find_unused_parameters,
        )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    use_bf16 = args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
    autocast = autocast_ctx(use_bf16)

    start_epoch, global_step, best_val = 0, 0, float("inf")
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        start_epoch, global_step, best_val = load_checkpoint(ckpt_path, policy, optimizer, device)
        D.print0(f"resumed from {ckpt_path} at epoch {start_epoch}, step {global_step}")
    D.barrier()

    effective_batch = args.batch_size * args.grad_accum * D.world_size()
    D.print0(
        f"dataset={args.dataset} episodes: {n_train_eps} train / {n_val_eps} val\n"
        f"lora_target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha}\n"
        f"params: total={n_total / 1e9:.2f}B trainable={n_trainable / 1e6:.2f}M "
        f"({100 * n_trainable / n_total:.3f}%)\n"
        f"world_size={D.world_size()} per_device_bs={args.batch_size} "
        f"grad_accum={args.grad_accum} effective_batch={effective_batch} bf16={use_bf16}"
    )

    run = None
    if args.wandb and D.is_main_process():
        import wandb

        run = wandb.init(
            project="vla-finetuning",
            dir=os.environ.get("WANDB_DIR"),
            config={
                **vars(args),
                "world_size": D.world_size(),
                "effective_batch": effective_batch,
                "trainable_params": n_trainable,
                "total_params": n_total,
            },
        )

    policy.train()
    stop = False
    # samples/s, not steps/s: grad_accum varies with world size in scaling runs
    timer = ThroughputTimer(device)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0, running, seen = time.time(), 0.0, 0
        for i, raw in enumerate(train_loader):
            batch = preprocessor(raw)
            is_accum = (i + 1) % args.grad_accum != 0

            # skip all-reduce on accumulation micro-batches
            use_no_sync = isinstance(policy, DDP) and is_accum
            sync_ctx = policy.no_sync if use_no_sync else contextlib.nullcontext
            with sync_ctx():
                with autocast():
                    loss, _ = policy(batch)
                (loss / args.grad_accum).backward()

            running += loss.item()
            seen += 1
            timer.add(args.batch_size * D.world_size())

            if not is_accum:
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == args.warmup_steps:
                    timer.start()

                if global_step % args.log_every == 0:
                    avg = running / max(seen, 1)
                    vram = D.peak_vram_gib(device)
                    D.print0(
                        f"epoch {epoch} step {global_step} loss {avg:.4f} peak_vram {vram:.1f}GiB"
                    )
                    if run is not None:
                        run.log(
                            {"train/loss": avg, "epoch": epoch, "vram_gib": vram}, step=global_step
                        )
                    running, seen = 0.0, 0

                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        val_loss = evaluate(policy, val_loader, preprocessor, autocast, device)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        D.print0(
            f"epoch {epoch} done in {time.time() - t0:.1f}s val_loss {val_loss:.4f}"
            f"{' (best)' if is_best else f' (best {best_val:.4f})'}"
        )
        if run is not None:
            run.log(
                {"val/loss": val_loss, "val/best_loss": best_val, "epoch": epoch}, step=global_step
            )

        save_checkpoint(
            args.out, ckpt_path, policy, optimizer, epoch + 1, global_step, best_val, is_best
        )
        D.barrier()
        if stop:
            D.print0(f"stopping at max_steps={args.max_steps}")
            break

    if timer.started:
        timer.pause()
        D.print0(
            f"THROUGHPUT world_size={D.world_size()} "
            f"per_device_bs={args.batch_size} grad_accum={args.grad_accum} "
            f"effective_batch={effective_batch} "
            f"samples_per_s={timer.samples_per_s:.1f} "
            f"elapsed_s={timer.elapsed:.1f} peak_vram_gib={D.peak_vram_gib(device):.1f}"
        )

    D.print0(f"best val_loss {best_val:.4f}; best adapter: {args.out / 'adapter_best'}")
    if run is not None:
        run.summary["val/best_loss"] = best_val
        run.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
