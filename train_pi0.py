"""LoRA fine-tuning of pi0 on LIBERO with native DDP. Runs as plain python or under torchrun."""

import argparse
import contextlib
import os
import shutil
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config

from src import dist_utils as D
from src.paths import run_dir


# PEFT fullmatches a string target_modules against the module path, hence the `.*` anchors.
# The vision tower is excluded on purpose: SigLIP features transfer well and are easy to damage.
_PROJECTIONS = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
_TOWERS = "gemma_expert|language_model"

LORA_TARGETS = {
    # lerobot default: action-expert q/v
    "default": rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({_PROJECTIONS}))",
    # q,k,v,o on both towers
    "attention": rf"(.*\.({_TOWERS})\..*\.self_attn\.(q|k|v|o)_proj|model\.({_PROJECTIONS}))",
    # attention + MLP on both towers
    "attention-mlp": (
        rf"(.*\.({_TOWERS})\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        rf"|model\.({_PROJECTIONS}))"
    ),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--policy", default="lerobot/pi0", help="base weights to adapt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="0 = no cap; for smoke tests")
    p.add_argument("--batch-size", type=int, default=8, help="per device")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4, help="default suits LoRA; full FT uses 2.5e-5")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-episodes", type=int, default=40, help="held out by episode, not by frame")
    p.add_argument(
        "--lora-target", choices=sorted(LORA_TARGETS), default="attention",
        help="which modules get adapters; see LORA_TARGETS",
    )
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument(
        "--out",
        type=Path,
        default=run_dir("pi0_libero"),
    )
    p.add_argument("--resume", action="store_true", help="resume from out/last.pt if present")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=5,
                   help="optimizer steps before throughput timing starts")
    p.add_argument("--no-find-unused-parameters", dest="find_unused_parameters",
                   action="store_false", default=True,
                   help="disable find_unused_parameters; only safe with --lora-target default")
    return p.parse_args()


def build_datasets(args, meta, cfg):
    """Split by episode: adjacent frames are near-duplicates."""
    # pi0 predicts action chunks
    delta_timestamps = resolve_delta_timestamps(cfg, meta)

    n_val = min(args.val_episodes, max(1, meta.total_episodes // 10))
    all_eps = list(range(meta.total_episodes))
    val_eps = all_eps[-n_val:]
    train_eps = all_eps[:-n_val]

    train_ds = LeRobotDataset(args.dataset, episodes=train_eps, delta_timestamps=delta_timestamps)
    val_ds = LeRobotDataset(args.dataset, episodes=val_eps, delta_timestamps=delta_timestamps)
    return train_ds, val_ds, len(train_eps), len(val_eps)


def build_loaders(args, train_ds, val_ds):
    train_sampler = val_sampler = None
    if D.is_distributed():
        # drop_last keeps ranks in lockstep
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
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
    return train_loader, val_loader, train_sampler


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


def save_checkpoint(args, path: Path, policy, optimizer, epoch: int, global_step: int,
                    best_val: float, is_best: bool):
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
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

    _export_adapter(raw, args.out / "adapter")
    if is_best:
        torch.save(payload, args.out / "best.pt.tmp")
        os.replace(args.out / "best.pt.tmp", args.out / "best.pt")
        _export_adapter(raw, args.out / "adapter_best")


def load_checkpoint(path: Path, policy, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    raw = policy.module if isinstance(policy, DDP) else policy
    missing, unexpected = raw.load_state_dict(ckpt["lora"], strict=False)
    if unexpected:
        raise RuntimeError(f"checkpoint has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    optimizer.load_state_dict(ckpt["optimizer"])
    # keep best_val across resume so a worse epoch cannot overwrite adapter_best
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_val", float("inf"))


@torch.no_grad()
def evaluate(policy, loader, preprocessor, autocast_ctx, device, seed: int = 1234):
    """Mean val loss across ranks.

    Stochastic (flow matching), so seeded per rank; RNG state is restored afterwards.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed + 10_000 * D.rank())

    policy.eval()
    loss_sum, n_seen = 0.0, 0
    for raw in loader:
        batch = preprocessor(raw)
        with autocast_ctx():
            loss, _ = policy(batch)
        bs = batch["action"].shape[0]
        loss_sum += loss.item() * bs
        n_seen += bs
    # reduce sums, not means: ranks may hold different sample counts
    total_loss = D.all_reduce_sum(loss_sum, device)
    total_n = D.all_reduce_sum(float(n_seen), device)
    policy.train()

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    return total_loss / max(total_n, 1.0)


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
    policy = policy.wrap_with_peft(peft_cli_overrides={
        "target_modules": LORA_TARGETS[args.lora_target],
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    })
    policy.to(device)

    preprocessor, _ = make_pre_post_processors(
        cfg, pretrained_path=args.policy,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

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
    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if use_bf16 else contextlib.nullcontext
    )

    start_epoch, global_step, best_val = 0, 0, float("inf")
    ckpt_path = args.out / "last.pt"
    if args.resume and ckpt_path.exists():
        start_epoch, global_step, best_val = load_checkpoint(
            ckpt_path, policy, optimizer, device
        )
        D.print0(f"resumed from {ckpt_path} at epoch {start_epoch}, step {global_step}")
    D.barrier()

    effective_batch = args.batch_size * args.grad_accum * D.world_size()
    D.print0(
        f"dataset={args.dataset} episodes: {n_train_eps} train / {n_val_eps} val\n"
        f"lora_target={args.lora_target} r={args.lora_r} alpha={args.lora_alpha}\n"
        f"params: total={n_total/1e9:.2f}B trainable={n_trainable/1e6:.2f}M "
        f"({100*n_trainable/n_total:.3f}%)\n"
        f"world_size={D.world_size()} per_device_bs={args.batch_size} "
        f"grad_accum={args.grad_accum} effective_batch={effective_batch} bf16={use_bf16}"
    )

    run = None
    if args.wandb and D.is_main_process():
        import wandb
        run = wandb.init(
            project="vla-finetuning", dir=os.environ.get("WANDB_DIR"),
            config={**vars(args), "world_size": D.world_size(),
                    "effective_batch": effective_batch,
                    "trainable_params": n_trainable, "total_params": n_total},
        )

    policy.train()
    stop = False
    # samples/s, not steps/s: grad_accum varies with world size in scaling runs
    timer_start, timed_samples = None, 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0, running, seen = time.time(), 0.0, 0
        for i, raw in enumerate(train_loader):
            batch = preprocessor(raw)
            is_accum = (i + 1) % args.grad_accum != 0

            # skip all-reduce on accumulation micro-batches
            sync_ctx = policy.no_sync if (isinstance(policy, DDP) and is_accum) else contextlib.nullcontext
            with sync_ctx():
                with autocast_ctx():
                    loss, _ = policy(batch)
                (loss / args.grad_accum).backward()

            running += loss.item()
            seen += 1
            if timer_start is not None:
                timed_samples += args.batch_size * D.world_size()

            if not is_accum:
                nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step == args.warmup_steps:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    D.barrier()
                    timer_start, timed_samples = time.time(), 0

                if global_step % args.log_every == 0:
                    avg = running / max(seen, 1)
                    vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
                    D.print0(
                        f"epoch {epoch} step {global_step} loss {avg:.4f} peak_vram {vram:.1f}GiB"
                    )
                    if run is not None:
                        run.log({"train/loss": avg, "epoch": epoch, "vram_gib": vram},
                                step=global_step)
                    running, seen = 0.0, 0

                if args.max_steps and global_step >= args.max_steps:
                    stop = True
                    break

        val_loss = evaluate(policy, val_loader, preprocessor, autocast_ctx, device)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        D.print0(
            f"epoch {epoch} done in {time.time() - t0:.1f}s val_loss {val_loss:.4f}"
            f"{' (best)' if is_best else f' (best {best_val:.4f})'}"
        )
        if run is not None:
            run.log({"val/loss": val_loss, "val/best_loss": best_val, "epoch": epoch},
                    step=global_step)

        save_checkpoint(args, ckpt_path, policy, optimizer, epoch + 1, global_step,
                        best_val, is_best)
        D.barrier()
        if stop:
            D.print0(f"stopping at max_steps={args.max_steps}")
            break

    if timer_start is not None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        D.barrier()
        elapsed = time.time() - timer_start
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        D.print0(
            f"THROUGHPUT world_size={D.world_size()} "
            f"per_device_bs={args.batch_size} grad_accum={args.grad_accum} "
            f"effective_batch={effective_batch} "
            f"samples_per_s={timed_samples / max(elapsed, 1e-9):.1f} "
            f"elapsed_s={elapsed:.1f} peak_vram_gib={peak:.1f}"
        )

    D.print0(
        f"best val_loss {best_val:.4f}; best adapter: {args.out / 'adapter_best'}"
    )
    if run is not None:
        run.summary["val/best_loss"] = best_val
        run.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
