"""Phase 2B: full fine-tuning of pi0 under FSDP, launched through Accelerate.

Why this file exists, given train_pi0.py already trains pi0 on LIBERO:

LoRA freezes the 4.03B base, so the optimizer allocates state for ~12M parameters and
the whole run fits in 11.7 GiB on a 40 GB card. Sharding would shard almost nothing —
shipping FSDP configs for that run would be untested decoration. *Full* fine-tuning is
where sharding becomes load-bearing:

    bf16 weights                 ~8 GiB
    bf16 gradients               ~8 GiB
    fp32 optimizer states (m,v) ~32 GiB
    fp32 master weights         ~16 GiB
    ------------------------------------
    before activations          ~64 GiB   -> OOMs a single A100-40GB

FSDP shards the last three across ranks. This script is the accelerate-launched
counterpart of train_pi0.py: same dataset, same episode-level split, same evaluate(),
imported from it rather than copied, so a full-FT number and a LoRA number are
comparable. train_pi0.py is left byte-identical — it is the verified production LoRA
recipe and its native-DDP loop is its own deliverable.

`--finetune lora` runs LoRA *under FSDP* as well; it exists so the memory table has a
like-for-like row, not to replace train_pi0.py, which is what produces the publishable
PEFT adapter.

### The wrap policy is not the usual one, and that is the interesting part

FSDP unshards parameters in a module's **pre-forward hook**, so a wrapped module only
works if its `forward` is actually called. pi0's PaliGemmaWithExpertModel does not call
its decoder layers: it hand-rolls a fused loop (`compute_layer_complete` in
lerobot/policies/pi0/modeling_pi0.py) that reaches into `layer.self_attn.q_proj(...)`
directly, so the VLM prefix and the action-expert suffix can share one attention op.
`GemmaDecoderLayer.forward` and `GemmaAttention.forward` are therefore dead code here.

Consequences:
  - the textbook `TRANSFORMER_BASED_WRAP` on `GemmaDecoderLayer` would create units whose
    hooks never fire, leaving parameters sharded when the math runs — wrong results or a
    shape error, not a clean failure;
  - we wrap `GemmaMLP` and `SiglipEncoderLayer` instead, which *are* invoked as modules
    (`layer.mlp(...)`, and the vision tower runs a stock HF forward). That covers ~2.4B
    of 4.03B parameters, and everything left over lives in the root unit;
  - DeepSpeed ZeRO-3 is unavailable for the same reason (it gathers parameters through
    module hooks too). ZeRO-2 would work, but FSDP's SHARD_GRAD_OP is the same algorithm
    without the extra dependency, so that is what we ship.
"""

import argparse
import contextlib
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Imports train_pi0 for the dataset split, the loaders and evaluate(). Importing it also
# runs preload_npp() at module level, which must happen before torchcodec is pulled in —
# scripts/eval_adapter.py depends on the same ordering.
import train_pi0 as T  # noqa: E402

import torch  # noqa: E402
from accelerate import Accelerator  # noqa: E402
from accelerate.utils import DistributedType, FullyShardedDataParallelPlugin  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402

from src import dist_utils as D  # noqa: E402
from src.paths import run_dir  # noqa: E402

# Matched by class *name* rather than by class object: no import from transformers'
# private module layout, and the policy stays readable in a log line. See the module
# docstring for why these two classes and not GemmaDecoderLayer.
FSDP_WRAP_CLASSES = ("GemmaMLP", "SiglipEncoderLayer")

# ShardingStrategy names, plus "ddp" for the unsharded baseline. The three FSDP entries
# are the ZeRO stages under their torch names:
#   no_shard       ~ ZeRO-0 / DDP, but through the FSDP code path
#   shard_grad_op  ~ ZeRO-2: gradients + optimizer state sharded, parameters replicated
#   full_shard     ~ ZeRO-3: parameters sharded too, re-gathered per unit on demand
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
    # pi0's own preset (PI0Config.optimizer_*). Full FT of a pretrained 4B model on 392
    # episodes is exactly where a LoRA-sized LR (1e-4) destroys the pretraining.
    p.add_argument("--lr", type=float, default=2.5e-5)
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
    # Gradients are reduced in fp32 by default: 4B parameters is enough accumulation for
    # bf16's 8-bit mantissa to visibly lose small updates. --reduce-dtype bf16 halves the
    # reduce-scatter traffic if the measurement says communication is the bottleneck.
    p.add_argument("--reduce-dtype", choices=("fp32", "bf16"), default="fp32")
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
    # save_state serialises the whole 4.03B model, LoRA or not — it is a *resume* point,
    # not an artefact. A benchmark row that never resumes should not pay 16 GiB of Lustre
    # writes for it.
    p.add_argument("--no-checkpoint", action="store_true",
                   help="skip the resumable checkpoint; benchmark rows only")
    # For the 1-GPU OOM demonstration: report the OOM as a result line and exit 0 rather
    # than failing the job. Only meaningful at world size 1 — a rank that dies mid-run
    # leaves the others hanging on the next collective.
    p.add_argument("--oom-ok", action="store_true")
    return p.parse_args()


def pi0_auto_wrap_policy(module, recurse: bool, nonwrapped_numel: int) -> bool:
    """FSDP auto-wrap callable: (module, recurse, nonwrapped_numel) -> bool.

    When `recurse` is True the return value means "descend into this module"; when it is
    False it means "make this module its own FSDP unit".
    """
    if recurse:
        return True
    return type(module).__name__ in FSDP_WRAP_CLASSES


def build_accelerator(args) -> Accelerator:
    common = dict(mixed_precision="bf16" if args.bf16 else "no")
    if args.sharding == "ddp":
        # The unsharded baseline. Goes through accelerate's plain DDP path so the only
        # variable against the FSDP rows of the table is the sharding strategy itself.
        return Accelerator(**common)

    reduce_dtype = torch.float32 if args.reduce_dtype == "fp32" else torch.bfloat16
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=1,
        reshard_after_forward=args.sharding.upper(),
        auto_wrap_policy=pi0_auto_wrap_policy,
        # The model is built in fp32 (see main): FSDP1 flattens each unit into one tensor
        # and refuses to mix dtypes inside it, and pi0's own bf16 mode deliberately keeps
        # the vision tower and the layernorms in fp32. So fp32 master weights + this
        # policy, which is the textbook mixed-precision setup anyway: bf16 for compute
        # and for the all-gather, fp32 for the master copy the optimizer updates.
        mixed_precision_policy={
            "param_dtype": torch.bfloat16,
            "reduce_dtype": reduce_dtype,
            "buffer_dtype": torch.float32,
        },
        # Required for mixed requires_grad inside one unit (LoRA), and it keeps
        # `.weight` a real per-parameter view, which pi0's fused loop reads directly.
        use_orig_params=True,
        backward_prefetch="BACKWARD_PRE",
        # Sharded checkpoints: no rank-0 gather of a 16 GiB state dict, and
        # torch.distributed.checkpoint reshards on load, so a run can resume onto a
        # different number of GPUs.
        state_dict_type="SHARDED_STATE_DICT",
        cpu_offload=args.cpu_offload,
        # Bounds the number of in-flight all-gathers; without it FSDP can queue enough
        # of them to defeat the point of sharding.
        limit_all_gathers=True,
        # pi0 applies its own torch.utils.checkpoint inside the fused layer loop
        # (--grad-checkpointing), so FSDP must not wrap the modules a second time.
        activation_checkpointing=False,
    )
    accelerator = Accelerator(fsdp_plugin=plugin, **common)
    # accelerate only promotes distributed_type to FSDP from an already-distributed
    # launch (accelerate/state.py: the FSDP branch sits inside the MULTI_GPU arm), so a
    # single-process run silently ignores the plugin and trains unsharded. That would
    # produce a plausible-looking row in the memory table that measures nothing. Fail
    # loudly instead: sharding across one rank shards nothing anyway, and the honest
    # single-GPU row is `--sharding ddp`.
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
    """Epoch/step/best-val bookkeeping, saved alongside model and optimizer.

    Registered with the Accelerator so `save_state`/`load_state` carry it; keeping it out
    of the checkpoint would make the first post-resume epoch look like an improvement and
    overwrite a genuinely better export.
    """

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
    """Sharded checkpoint written beside the target, then swapped in.

    Every rank writes its own shard, so the swap has to happen after all of them are
    done, and only on rank 0. A half-written checkpoint on Lustre is worse than none.
    """
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
    """Gather the sharded weights and write a directory PI0Policy.from_pretrained loads.

    `get_state_dict` runs a cross-rank collective, so *every* rank must call it — only the
    writing is rank-gated. lerobot's `save_pretrained(state_dict=...)` exists precisely
    for this case; calling it without the gathered dict would serialise FSDP shards.
    """
    if args.no_export:
        return
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
            # LoRA under FSDP is a benchmark row, not a publishable artefact: train_pi0.py
            # is what writes a real PEFT adapter directory. Save just the trained tensors.
            torch.save({k: v for k, v in state.items() if "lora" in k.lower()},
                       tmp / "lora.pt")
        else:
            unwrapped.save_pretrained(tmp, state_dict=state)
            # The processors carry the dataset normalisation statistics; without them the
            # exported weights load but predict nonsense.
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
        # fp32 always: FSDP1 refuses to flatten mixed dtypes into one unit, and pi0's
        # bf16 mode is deliberately mixed (fp32 vision tower and layernorms). bf16 for
        # compute comes from the FSDP mixed-precision policy plus accelerate's autocast.
        dtype="float32",
        gradient_checkpointing=args.grad_checkpointing,
    )

    # Identical split and loaders to the DDP LoRA run, imported rather than reimplemented:
    # a divergent copy would silently validate on different episodes and the two runs'
    # val losses would not be comparable.
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
    # How much of the model the wrap policy actually reaches. If this collapses to ~0 the
    # class names have drifted with a transformers upgrade and every parameter would fall
    # back into the root unit — FULL_SHARD would then behave exactly like SHARD_GRAD_OP
    # and the memory table would be quietly wrong rather than visibly broken.
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
    # Built after prepare(): under FSDP the parameters the optimizer must own are the
    # sharded ones, and with use_orig_params=False they would not even exist yet.
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay,
    )
    optimizer = accelerator.prepare(optimizer)

    # pi0 is built in fp32 and materialised on the GPU *before* FSDP shards it, which costs
    # ~16.1 GiB transiently. `max_memory_allocated` is a process-lifetime high-water mark, so
    # without this reset that transient floors every reported peak and hides any difference
    # between sharding strategies below it — which is exactly what happened at world size 8,
    # where ZeRO-2 and ZeRO-3 both read 17.0 GiB. Record it, then reset so the reported
    # training peak measures training.
    setup_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    steps_per_epoch = max(len(train_loader) // args.grad_accum, 1)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, args, total_steps)
    )
    # Deliberately not passed through accelerate.prepare: an AcceleratedScheduler steps
    # once per *process* per optimizer step, which would silently multiply the schedule
    # by world_size. Stepping it by hand keeps the curve identical at every world size.
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
        # The single-GPU baseline is *expected* to die here — that is the measurement.
        # Reported as a result line so a sweep can tabulate it without parsing a traceback.
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
    # Throughput is timed as the sum of the *training* segments only: one segment per
    # epoch, opened when the loop starts stepping and closed before validation. A single
    # start-to-finish stopwatch would fold every epoch's validation pass and its 16 GiB
    # checkpoint write into the window, which understates samples/s by roughly 10% on a
    # five-epoch run and by far more on a 30-step benchmark row.
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
            # No no_sync() on the accumulation micro-batches, unlike the DDP loop in
            # train_pi0.py: FSDP's no_sync keeps *unsharded* gradients resident for the
            # whole window, which gives back exactly the memory sharding just saved.
            # Reduce-scattering every micro-batch accumulates into the sharded grad and
            # costs communication we are explicitly trading memory for.
            accelerator.backward(loss / args.grad_accum)

            running += loss.item()
            seen += 1
            if timed_started:
                timed_samples += args.batch_size * accelerator.num_processes

            if (i + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    # Routes to FSDP's own clip_grad_norm_, which reduces the norm across
                    # shards; torch.nn.utils.clip_grad_norm_ would clip on each rank's
                    # slice and produce a different, smaller global norm.
                    accelerator.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step == args.warmup_steps and not timed_started:
                    # Barrier first, so the clock starts with every rank at the same
                    # point and a straggler's startup does not land inside the window.
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    accelerator.wait_for_everyone()
                    timed_started, seg_start, timed_samples = True, time.time(), 0

                if state.global_step % args.log_every == 0:
                    avg = running / max(seen, 1)
                    vram = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
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
            # Peak is read here too: the export gathers full parameters through the GPU,
            # and letting that land in the reported number would overstate what training
            # itself costs.
            timed_peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0

        # nullcontext: accelerate already wrapped the model's forward in autocast, so a
        # second one here would only nest.
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
        # One machine-greppable line per run, same shape as train_pi0.py's, so the
        # sharding sweep and the LoRA scaling sweep tabulate with the same parser.
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
