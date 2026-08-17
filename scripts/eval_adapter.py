"""Reload a saved LoRA adapter onto base pi0 and re-measure validation loss.

This is the check that the *published* artefact works, not just the in-memory model at
the end of training: it loads `lerobot/pi0` fresh, applies the adapter directory, and
evaluates on the same held-out episodes training used. If the number does not match the
training log, the adapter is not what it claims to be.

Deliberately imports the training module rather than reimplementing the split and the
eval loop — a divergent copy here would compare against the wrong episodes and quietly
report a number nobody can reproduce.

    srun -p plgrid-now -A <account> --gres=gpu:1 --cpus-per-task=16 --mem=100G \
        -t 00:30:00 python scripts/eval_adapter.py --adapter <run>/adapter_best
"""

import argparse
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_pi0 as T  # noqa: E402  (runs preload_npp before torchcodec is imported)
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True, help="PEFT adapter directory")
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--policy", default="lerobot/pi0")
    # Must match training, or the "validation" set is a different set of episodes.
    p.add_argument("--val-episodes", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    # pi0 is a flow-matching model: every forward samples a diffusion timestep and noise,
    # so validation loss is a *stochastic* estimate, not a fixed number. Without seeding,
    # repeated evaluation of the same weights gives different answers. Repeat and report
    # spread, so epoch-to-epoch differences can be judged against the noise floor.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(args.dataset)
    cfg = PI0Config(
        pretrained_path=args.policy, device=device.type,
        dtype="bfloat16" if args.bf16 else "float32",
    )

    ds_args = SimpleNamespace(dataset=args.dataset, val_episodes=args.val_episodes)
    _, val_ds, n_train_eps, n_val_eps = T.build_datasets(ds_args, meta, cfg)
    print(f"{args.dataset}: {n_train_eps} train / {n_val_eps} val episodes "
          f"({len(val_ds)} val frames)")

    policy = make_policy(cfg, ds_meta=meta)
    from peft import PeftModel
    policy = PeftModel.from_pretrained(policy, str(args.adapter))
    n_lora = sum(p_.numel() for n_, p_ in policy.named_parameters() if "lora" in n_.lower())
    print(f"adapter {args.adapter}: {n_lora/1e6:.2f}M LoRA params applied")

    policy.to(device).eval()
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

    autocast_ctx = (
        (lambda: torch.autocast("cuda", dtype=torch.bfloat16))
        if args.bf16 and device.type == "cuda" else contextlib.nullcontext
    )
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    losses = []
    for i in range(args.repeats):
        torch.manual_seed(args.seed + i)
        losses.append(T.evaluate(policy, loader, preprocessor, autocast_ctx, device))
        print(f"  repeat {i} (seed {args.seed + i}): {losses[-1]:.4f}")

    mean = sum(losses) / len(losses)
    spread = (max(losses) - min(losses)) if len(losses) > 1 else 0.0
    std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5 if len(losses) > 1 else 0.0
    print(f"\nVAL_LOSS mean={mean:.4f} std={std:.4f} spread={spread:.4f} "
          f"n={len(losses)} adapter={args.adapter}")


if __name__ == "__main__":
    main()
