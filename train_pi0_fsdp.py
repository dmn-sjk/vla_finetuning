"""Full fine-tuning of pi0 under FSDP via accelerate (--finetune lora for comparison).

pi0 never calls its decoder layers' forward, so the wrap policy targets GemmaMLP and
SiglipEncoderLayer instead of GemmaDecoderLayer. See docs/notes.md.
"""

import argparse
import contextlib
import math
import os
import shutil
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config

import train_pi0 as T
from src import dist_utils as D
from src.paths import run_dir

# matched by class name to avoid importing transformers internals
FSDP_WRAP_CLASSES = ("GemmaMLP", "SiglipEncoderLayer")

# "ddp" is the unsharded baseline; the rest are torch ShardingStrategy names
SHARDING_CHOICES = ("ddp", "no_shard", "shard_grad_op", "full_shard")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--policy", default="lerobot/pi0")
    p.add_argument("--finetune", choices=("full", "lora"), default="full")
    p.add_argument("--sharding", choices=SHARDING_CHOICES, default="full_shard")
    p.add_argument("--cpu-offload", action="store_true",
                   help="offload sharded params+grads to host RAM; last resort, very slow")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="0 = no cap; for smoke tests")
    p.add_argument("--batch-size", type=int, default=8, help="per device")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2.5e-5,
                   help="pi0's own preset; a LoRA-sized LR destroys the pretraining")
    p.add_argument("--lr-min", type=float, default=2.5e-6, help="cosine floor")
    p.add_argument("--lr-warmup-steps", type=int, default=200)
    p.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-episodes", type=int, default=40)
    p.add_argument("--lora-target", choices=sorted(T.LORA_TARGETS), default="attention")
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--reduce-dtype", choices=("fp32", "bf16"), default="fp32",
                   help="gradient reduce dtype; bf16 halves traffic but loses small updates")
    p.add_argument(
        "--out", type=Path,
        default=run_dir("pi0_full"),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--warmup-steps", type=int, default=5,
                   help="optimizer steps to skip before the throughput timer starts")
    p.add_argument("--export-dtype", choices=("bfloat16", "float32"), default="bfloat16",
                   help="dtype of the exported full-FT checkpoint (16 GiB in fp32)")
    p.add_argument("--no-export", action="store_true",
                   help="skip full-model export; for benchmark runs, where it is dead I/O")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="skip the 16 GiB resumable checkpoint; benchmark rows only")
    p.add_argument("--oom-ok", action="store_true",
                   help="report OOM as a result line and exit 0; world size 1 only")
    return p.parse_args()


def pi0_auto_wrap_policy(module, recurse: bool, nonwrapped_numel: int) -> bool:
    """FSDP auto-wrap policy: always recurse; wrap modules named in FSDP_WRAP_CLASSES."""
    if recurse:
        return True
    return type(module).__name__ in FSDP_WRAP_CLASSES


def build_accelerator(args) -> Accelerator:
    common = dict(mixed_precision="bf16" if args.bf16 else "no")
    if args.sharding == "ddp":
        return Accelerator(**common)

    reduce_dtype = torch.float32 if args.reduce_dtype == "fp32" else torch.bfloat16
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=1,
        reshard_after_forward=args.sharding.upper(),
        auto_wrap_policy=pi0_auto_wrap_policy,
        # fp32 model (FSDP1 cannot flatten mixed dtypes), bf16 compute and all-gather
        mixed_precision_policy={
            "param_dtype": torch.bfloat16,
            "reduce_dtype": reduce_dtype,
            "buffer_dtype": torch.float32,
        },
        # needed for mixed requires_grad (LoRA) and for pi0 reading .weight directly
        use_orig_params=True,
        backward_prefetch="BACKWARD_PRE",
        state_dict_type="SHARDED_STATE_DICT",
        cpu_offload=args.cpu_offload,
        limit_all_gathers=True,
        # pi0 does its own activation checkpointing
        activation_checkpointing=False,
    )
    accelerator = Accelerator(fsdp_plugin=plugin, **common)
    # accelerate ignores the FSDP plugin at world size 1
    if accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            f"--sharding {args.sharding} requires a distributed launch with world size > 1 "
            f"(got distributed_type={accelerator.distributed_type}, "
            f"world_size={accelerator.num_processes}). Use --sharding ddp on one GPU."
        )
    return accelerator


def lr_lambda(step: int, args, total_steps: int) -> float:
    """Multiplier on args.lr. Linear warmup, then cosine to args.lr_min."""
    if step < args.lr_warmup_steps:
        return (step + 1) / max(args.lr_warmup_steps, 1)
    if args.lr_schedule == "constant":
        return 1.0
    progress = (step - args.lr_warmup_steps) / max(total_steps - args.lr_warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    floor = args.lr_min / args.lr
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


class TrainState:
    """Epoch/step/best-val bookkeeping, checkpointed with the model so resume keeps best_val."""

    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_val = float("inf")

    def state_dict(self):
        return {"epoch": self.epoch, "global_step": self.global_step, "best_val": self.best_val}

    def load_state_dict(self, sd):
        self.epoch = sd["epoch"]
        self.global_step = sd["global_step"]
        self.best_val = sd["best_val"]


def save_state(accelerator, args, path: Path):
    """Sharded checkpoint written beside the target, then swapped in by rank 0."""
    tmp = path.with_name(path.name + ".tmp")
    if accelerator.is_main_process:
        shutil.rmtree(tmp, ignore_errors=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(tmp))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    accelerator.wait_for_everyone()


def export_model(accelerator, args, policy, preprocessor, postprocessor, dest: Path):
    """Gather the sharded weights and write a directory PI0Policy.from_pretrained loads."""
    if args.no_export:
        return
    # get_state_dict is a collective: every rank must call it
    state = accelerator.get_state_dict(policy)
    if accelerator.is_main_process:
        if args.export_dtype == "bfloat16":
            state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                     for k, v in state.items()}
        tmp = dest.with_name(dest.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(policy)
        if args.finetune == "lora":
            # benchmark row only; train_pi0.py writes the real PEFT adapter
            torch.save({k: v for k, v in state.items() if "lora" in k.lower()},
                       tmp / "lora.pt")
        else:
            unwrapped.save_pretrained(tmp, state_dict=state)
            # processors carry the normalization stats
            preprocessor.save_pretrained(tmp)
            postprocessor.save_pretrained(tmp)
        shutil.rmtree(dest, ignore_errors=True)
        os.replace(tmp, dest)
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    accelerator = build_accelerator(args)   # also initialises the process group
    device = accelerator.device
    torch.manual_seed(args.seed)

    meta = LeRobotDatasetMetadata(args.dataset)
    cfg = PI0Config(
        pretrained_path=args.policy,
        device=device.type,
        # fp32 always; bf16 compute comes from the FSDP mixed-precision policy
        dtype="float32",
        gradient_checkpointing=args.grad_checkpointing,
    )

    train_ds, val_ds, n_train_eps, n_val_eps = T.build_datasets(args, meta, cfg)
    train_loader, val_loader, train_sampler = T.build_loaders(args, train_ds, val_ds)

    policy = make_policy(cfg, ds_meta=meta)
    if args.finetune == "lora":
        policy = policy.wrap_with_peft(peft_cli_overrides={
            "target_modules": T.LORA_TARGETS[args.lora_target],
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        })

    preprocessor, postprocessor = make_pre_post_processors(
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

    n_total = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    # guard against class renames collapsing every param into the root unit
    n_wrapped = sum(
        sum(p.numel() for p in m.parameters())
        for m in policy.modules() if type(m).__name__ in FSDP_WRAP_CLASSES
    )
    if args.sharding != "ddp" and n_wrapped < 0.5 * n_total:
        raise RuntimeError(
            f"auto-wrap policy covers only {n_wrapped/1e9:.2f}B of {n_total/1e9:.2f}B "
            f"parameters via {FSDP_WRAP_CLASSES}; the class names have probably moved"
        )

    policy = accelerator.prepare(policy)
    # built after prepare() so the optimizer owns the sharded parameters
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay,
    )
    optimizer = accelerator.prepare(optimizer)

    # exclude the fp32 construction transient from the training peak
    setup_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args, total_steps)
    )
    # stepped manually: AcceleratedScheduler steps once per process
    state = TrainState()
    accelerator.register_for_checkpointing(state, scheduler)

    ckpt_dir = args.out / "state"
    if args.resume and ckpt_dir.exists():
        accelerator.load_state(str(ckpt_dir))
        D.print0(f"resumed from {ckpt_dir} at epoch {state.epoch}, step {state.global_step}")
    accelerator.wait_for_everyone()

    effective_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    D.print0(
        f"dataset={args.dataset} episodes: {n_train_eps} train / {n_val_eps} val\n"
        f"finetune={args.finetune} sharding={args.sharding} "
        f"cpu_offload={args.cpu_offload} grad_checkpointing={args.grad_checkpointing}\n"
        f"params: total={n_total/1e9:.2f}B trainable={n_trainable/1e9:.3f}B "
        f"({100*n_trainable/n_total:.2f}%) fsdp_wrapped={n_wrapped/1e9:.2f}B "
        f"({100*n_wrapped/max(n_total,1):.0f}% of the model)\n"
        f"world_size={accelerator.num_processes} per_device_bs={args.batch_size} "
        f"grad_accum={args.grad_accum} effective_batch={effective_batch} "
        f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} bf16={args.bf16}"
    )

    run = None
    if args.wandb and accelerator.is_main_process:
        import wandb
        run = wandb.init(
            project="vla-finetuning", dir=os.environ.get("WANDB_DIR"),
            config={**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    "world_size": accelerator.num_processes,
                    "effective_batch": effective_batch,
                    "trainable_params": n_trainable, "total_params": n_total},
        )

    try:
        train(args, accelerator, policy, optimizer, scheduler, state, preprocessor,
              postprocessor, train_loader, val_loader, train_sampler, effective_batch,
              ckpt_dir, run, device, setup_peak)
    except torch.OutOfMemoryError:
        if not args.oom_ok:
            raise
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        D.print0(
            f"OOM_EXPECTED finetune={args.finetune} sharding={args.sharding} "
            f"world_size={accelerator.num_processes} per_device_bs={args.batch_size} "
            f"grad_checkpointing={args.grad_checkpointing} "
            f"peak_vram_gib={peak:.1f} capacity_gib="
            f"{torch.cuda.get_device_properties(device).total_memory / 2**30:.1f}"
        )

    if run is not None:
        run.summary["val/best_loss"] = state.best_val
        run.finish()
    accelerator.end_training()


def train(args, accelerator, policy, optimizer, scheduler, state, preprocessor,
          postprocessor, train_loader, val_loader, train_sampler, effective_batch,
          ckpt_dir, run, device, setup_peak=0.0):
    policy.train()
    stop = False
    # time training segments only, excluding val and checkpoint writes
    timed_started, seg_start, timed_elapsed, timed_samples, timed_peak = False, None, 0.0, 0, 0.0
    trainable = [p for p in policy.parameters() if p.requires_grad]

    for epoch in range(state.epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if timed_started and seg_start is None:
            seg_start = time.time()

        t0, running, seen = time.time(), 0.0, 0
        for i, raw in enumerate(train_loader):
            batch = preprocessor(raw)
            with accelerator.autocast():
                loss, _ = policy(batch)
            # no no_sync: it keeps unsharded grads resident
            accelerator.backward(loss / args.grad_accum)

            running += loss.item()
            seen += 1
            if timed_started:
                timed_samples += args.batch_size * accelerator.num_processes

            if (i + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    # accelerate routes to FSDP's sharded clip_grad_norm_
                    accelerator.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step == args.warmup_steps and not timed_started:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    accelerator.wait_for_everyone()
                    timed_started, seg_start, timed_samples = True, time.time(), 0

                if state.global_step % args.log_every == 0:
                    avg = running / max(seen, 1)
                    cuda = device.type == "cuda"
                    vram = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
                    D.print0(
                        f"epoch {epoch} step {state.global_step} loss {avg:.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e} peak_vram {vram:.1f}GiB"
                    )
                    if run is not None:
                        run.log({"train/loss": avg, "epoch": epoch, "vram_gib": vram,
                                 "lr": scheduler.get_last_lr()[0]}, step=state.global_step)
                    running, seen = 0.0, 0

                if args.max_steps and state.global_step >= args.max_steps:
                    stop = True
                    break

        if seg_start is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            accelerator.wait_for_everyone()
            timed_elapsed += time.time() - seg_start
            seg_start = None
            # read before export, which gathers full parameters on the GPU
            timed_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

        # accelerate already wraps the forward in autocast
        val_loss = T.evaluate(policy, val_loader, preprocessor, contextlib.nullcontext, device)
        is_best = val_loss < state.best_val
        state.best_val = min(state.best_val, val_loss)
        state.epoch = epoch + 1
        D.print0(
            f"epoch {epoch} done in {time.time() - t0:.1f}s val_loss {val_loss:.4f}"
            f"{' (best)' if is_best else f' (best {state.best_val:.4f})'}"
        )
        if run is not None:
            run.log({"val/loss": val_loss, "val/best_loss": state.best_val, "epoch": epoch},
                    step=state.global_step)

        if not args.no_checkpoint:
            save_state(accelerator, args, ckpt_dir)
        if is_best:
            export_model(accelerator, args, policy, preprocessor, postprocessor,
                         args.out / "model_best")
        if stop:
            D.print0(f"stopping at max_steps={args.max_steps}")
            break

    if timed_started and timed_elapsed > 0:
        elapsed, peak = timed_elapsed, timed_peak
        D.print0(
            f"THROUGHPUT finetune={args.finetune} sharding={args.sharding} "
            f"world_size={accelerator.num_processes} per_device_bs={args.batch_size} "
            f"grad_accum={args.grad_accum} effective_batch={effective_batch} "
            f"grad_checkpointing={args.grad_checkpointing} "
            f"samples_per_s={timed_samples / max(elapsed, 1e-9):.1f} "
            f"elapsed_s={elapsed:.1f} peak_vram_gib={peak:.1f} "
            f"setup_peak_gib={setup_peak:.1f}"
        )

    export_model(accelerator, args, policy, preprocessor, postprocessor, args.out / "model")
    D.print0(
        f"best val_loss {state.best_val:.4f}; "
        f"{args.out / 'model_best'} is the checkpoint to publish"
    )


if __name__ == "__main__":
    main()
