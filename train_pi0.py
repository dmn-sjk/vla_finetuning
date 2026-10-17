"""Phase 2: LoRA fine-tuning of pi0 on LIBERO, native torch.distributed.

Runs identically at world size 1 (`python train_pi0.py`) and under torchrun, using the
distributed scaffolding in src/dist_utils.

What it does:
  - the model is pi0 (~4B) from lerobot, frozen except for LoRA adapters
  - batches come from LeRobotDataset and go through lerobot's processor pipeline
  - checkpoints save the *adapter*, not the base weights, since only the adapter trains
"""

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


# pi0's Linear modules fall into four groups (names dumped from the loaded model):
#
#   model.paligemma_with_expert.paligemma.model.language_model.layers.N.{self_attn,mlp}.*
#   model.paligemma_with_expert.gemma_expert.model.layers.N.{self_attn,mlp}.*
#   model.paligemma_with_expert.paligemma.model.vision_tower...encoder.layers.N.*
#   model.{state_proj,action_in_proj,action_out_proj,action_time_mlp_in,action_time_mlp_out}
#
# PEFT treats a string `target_modules` as a regex matched with re.fullmatch against
# the module path, which is why every pattern here is anchored with `.*`.
#
# The vision tower is deliberately excluded from all presets: SigLIP features transfer
# well across manipulation tasks, and adapting them on 428 episodes is the fastest way
# to damage the part of the model doing the most generalisation work.
_PROJECTIONS = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
_TOWERS = "gemma_expert|language_model"

LORA_TARGETS = {
    # lerobot's own default: action expert q/v only. ~0.7M params at r=8 (0.017%) —
    # kept so runs against the library default remain reproducible.
    "default": rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({_PROJECTIONS}))",
    # Recommended: full attention (q,k,v,o) on BOTH the VLM backbone and the action
    # expert. ~12.7M at r=32 (0.31%), a normal LoRA operating point.
    "attention": rf"(.*\.({_TOWERS})\..*\.self_attn\.(q|k|v|o)_proj|model\.({_PROJECTIONS}))",
    # Aggressive: attention + MLP on both towers. ~53M at r=32 (1.3%). More capacity,
    # more overfitting risk on a single 428-episode suite.
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
    p.add_argument("--lr", type=float, default=1e-4, help="LoRA tolerates more than the 2.5e-5 full-FT default")
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
    # bf16 on by default (A100 supports it); --no-bf16 falls back for debugging numerics.
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument(
        "--out",
        type=Path,
        default=run_dir("pi0_libero"),
    )
    p.add_argument("--resume", action="store_true", help="resume from out/last.pt if present")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    # Throughput timing starts only after this many optimizer steps, so model load,
    # cuDNN autotuning and the first allocator growth don't pollute the measurement.
    p.add_argument("--warmup-steps", type=int, default=5)
    # Required whenever LoRA is attached to the VLM tower (`attention`, `attention-mlp`):
    # pi0 consumes only the prefix KV cache from PaliGemma, so the *final* language-model
    # layer's o_proj output feeds nothing and its adapter receives no gradient. DDP then
    # aborts with "Expected to have finished reduction in the prior iteration".
    # Measured: exactly 2 of 298 trainable tensors are affected.
    # Safe to disable for --lora-target default, which adapts only the action expert.
    p.add_argument("--no-find-unused-parameters", dest="find_unused_parameters",
                   action="store_false", default=True)
    return p.parse_args()


def build_datasets(args, meta, cfg):
    """Split by *episode*, not by frame.

    Consecutive frames in a trajectory are near-duplicates, so a frame-level split
    leaks almost the whole validation set into training and the val curve becomes
    meaningless. Holding out whole episodes is the only honest split here.
    """
    # pi0 predicts an action chunk, so the dataset must return a window of future
    # actions. lerobot derives the window from the config's delta indices.
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
        # drop_last keeps every rank's epoch the same length, so no rank runs out of
        # batches early and deadlocks the others at the next collective.
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
    """Only the LoRA parameters — the base 4B is frozen and already on the Hub."""
    return {k: v.detach().cpu() for k, v in policy.state_dict().items() if "lora" in k.lower()}


def _export_adapter(raw, dest: Path):
    """Write a PEFT adapter directory, swapping it in only once complete.

    `save_pretrained` writes several files, so a crash midway leaves a directory that
    looks loadable but isn't. Build beside the target, then rename.
    """
    tmp = dest.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    raw.save_pretrained(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)


def save_checkpoint(args, path: Path, policy, optimizer, epoch: int, global_step: int,
                    best_val: float, is_best: bool):
    """Rank-0 only, written atomically.

    A torn checkpoint on Lustre is worse than no checkpoint, and long runs mean we will
    need to resume at least once.

    Two artefacts, because they answer different questions:
      - `last.pt` + `adapter/`      -> resume the run where it stopped
      - `best.pt` + `adapter_best/` -> the adapter that actually generalises best
    pi0+LoRA overfits LIBERO-spatial after ~5 epochs, so the final epoch is routinely
    *worse* than the best one. Publishing `adapter/` would ship the wrong weights.
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

    # PEFT's own layout — what gets pushed to the Hub (deliverable #2) and what
    # `PeftModel.from_pretrained` expects.
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
    # Carry best_val across a resume, or the first post-resume epoch always looks like
    # an improvement and overwrites a genuinely better adapter.
    return ckpt["epoch"], ckpt["global_step"], ckpt.get("best_val", float("inf"))


@torch.no_grad()
def evaluate(policy, loader, preprocessor, autocast_ctx, device, seed: int = 1234):
    """Validation loss. Note this is a *stochastic* estimate, not a fixed property of
    the weights: pi0 is a flow-matching model, so every forward samples a diffusion
    timestep and a noise vector.

    The per-rank seed offset matters. Every rank starts from the same
    `torch.manual_seed(args.seed)` and executes the same op sequence, so without an
    offset all ranks draw the *same* timestep sequence. That correlates the noise
    across ranks, cutting the number of independent draws by a factor of world_size
    and inflating the variance of the reduced estimate by roughly sqrt(world_size)
    instead of averaging it down. Measured effect: training-time val loss scattered
    +-0.007 around the true value (0.3343 +- 0.0015 from a seeded single-process eval),
    which is large enough to reorder adjacent epochs and pick the wrong "best".

    RNG state is saved and restored so seeding here cannot perturb training.
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
    # Reduce sums, not means: ranks may hold different sample counts.
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

    # Same seed on every rank so LoRA init is identical; the sampler, not the seed,
    # is what makes ranks see different data.
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
    # A target_modules regex that matches nothing still "succeeds" and silently yields
    # an adapter that trains almost nothing — assert instead of discovering it in the
    # loss curve two hours later.
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
    # Only LoRA params are passed to the optimizer, so no optimizer state is
    # allocated for the frozen 4B — that is most of the memory saving.
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
    # Throughput is measured in samples/s across all ranks, not steps/s: when the
    # effective batch is held constant while scaling GPUs, grad_accum changes
    # inversely, so steps/s is not comparable between world sizes but samples/s is.
    timer_start, timed_samples = None, 0

    for epoch in range(start_epoch, args.epochs):
        # Without this, every rank reshuffles identically each epoch and the sharding
        # silently degrades. Costs one line, invisible on 1 GPU.
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0, running, seen = time.time(), 0.0, 0
        for i, raw in enumerate(train_loader):
            batch = preprocessor(raw)
            is_accum = (i + 1) % args.grad_accum != 0

            # Skip the gradient all-reduce on micro-batches that are not the last of
            # their accumulation window — this is where most of the speedup from
            # gradient accumulation actually comes from.
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
                    # Barrier first so the clock starts with every rank at the same
                    # point; otherwise a straggler's startup lands inside the window.
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    D.barrier()
                    timer_start, timed_samples = time.time(), 0

                if global_step % args.log_every == 0:
                    avg = running / max(seen, 1)
                    # Peak VRAM in the progress line, not just W&B: on 40 GB cards the
                    # headroom is the number you actually need while tuning batch size.
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
        # Hold every rank until the write lands, so a crash mid-save cannot leave
        # ranks disagreeing about what the last good checkpoint is.
        D.barrier()
        if stop:
            D.print0(f"stopping at max_steps={args.max_steps}")
            break

    # One machine-greppable line so a scaling sweep can be tabulated without parsing
    # the whole log. Printed even when the run was cut short by --max-steps.
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
        f"best val_loss {best_val:.4f}; publish {args.out / 'adapter_best'} "
        f"(NOT {args.out / 'adapter'}, which is the final epoch)"
    )
    if run is not None:
        run.summary["val/best_loss"] = best_val
        run.finish()
    D.cleanup()


if __name__ == "__main__":
    main()
