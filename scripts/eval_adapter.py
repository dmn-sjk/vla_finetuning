"""Reload a saved checkpoint onto pi0 and re-measure validation loss.

Loads the base weights fresh, applies the checkpoint, and evaluates on the training split's
held-out episodes, so the result is comparable to the training log.

    python scripts/eval_adapter.py --adapter <run>/adapter_best --repeats 5
    python scripts/eval_adapter.py --policy <run>/model_best --repeats 5
"""

import argparse
import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import train_pi0 as T  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="PEFT adapter directory; omit to evaluate --policy on its own",
    )
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument(
        "--policy",
        default="lerobot/pi0",
        help="base weights, or a full fine-tuned checkpoint directory",
    )
    p.add_argument("--val-episodes", type=int, default=40, help="must match training")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="repeat with seeds seed..seed+N-1 and report mean/std",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = LeRobotDatasetMetadata(args.dataset)
    cfg = PI0Config(
        pretrained_path=args.policy,
        device=device.type,
        dtype="bfloat16" if args.bf16 else "float32",
    )

    ds_args = SimpleNamespace(dataset=args.dataset, val_episodes=args.val_episodes)
    _, val_ds, n_train_eps, n_val_eps = T.build_datasets(ds_args, meta, cfg)
    print(
        f"{args.dataset}: {n_train_eps} train / {n_val_eps} val episodes ({len(val_ds)} val frames)"
    )

    policy = make_policy(cfg, ds_meta=meta)
    if args.adapter is not None:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, str(args.adapter))
        n_lora = sum(p_.numel() for n_, p_ in policy.named_parameters() if "lora" in n_.lower())
        print(f"adapter {args.adapter}: {n_lora / 1e6:.2f}M LoRA params applied")
    else:
        print(
            f"policy {args.policy}: "
            f"{sum(p_.numel() for p_ in policy.parameters()) / 1e9:.2f}B params, no adapter"
        )

    policy.to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        cfg,
        pretrained_path=args.policy,
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
        if args.bf16 and device.type == "cuda"
        else contextlib.nullcontext
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    losses = []
    for i in range(args.repeats):
        losses.append(
            T.evaluate(policy, loader, preprocessor, autocast_ctx, device, seed=args.seed + i)
        )
        print(f"  repeat {i} (seed {args.seed + i}): {losses[-1]:.4f}")

    mean = sum(losses) / len(losses)
    spread = (max(losses) - min(losses)) if len(losses) > 1 else 0.0
    std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5 if len(losses) > 1 else 0.0
    print(
        f"\nVAL_LOSS mean={mean:.4f} std={std:.4f} spread={spread:.4f} "
        f"n={len(losses)} checkpoint={args.adapter or args.policy}"
    )


if __name__ == "__main__":
    main()
