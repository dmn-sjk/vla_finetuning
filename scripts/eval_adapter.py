"""Reload a saved checkpoint onto pi0 and re-measure validation loss.

This is the check that the *published* artefact works, not just the in-memory model at
the end of training: it loads the base weights fresh, applies the saved checkpoint, and
evaluates on the same held-out episodes training used. If the number does not match the
training log, the artefact is not what it claims to be.

Two shapes of artefact, one eval path, so the numbers are comparable:

    # LoRA adapter from train_pi0.py
    python scripts/eval_adapter.py --adapter <run>/adapter_best --repeats 5
    # full fine-tune exported by train_pi0_fsdp.py (a complete pi0 checkpoint)
    python scripts/eval_adapter.py --policy <run>/model_best --repeats 5

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
    p.add_argument("--adapter", type=Path, default=None,
                   help="PEFT adapter directory; omit to evaluate --policy on its own")
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--policy", default="lerobot/pi0",
                   help="base weights, or a full fine-tuned checkpoint directory")
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
    if args.adapter is not None:
        from peft import PeftModel
        policy = PeftModel.from_pretrained(policy, str(args.adapter))
        n_lora = sum(p_.numel() for n_, p_ in policy.named_parameters() if "lora" in n_.lower())
        print(f"adapter {args.adapter}: {n_lora/1e6:.2f}M LoRA params applied")
    else:
        # A full fine-tune: the weights came from --policy, there is nothing to apply.
        print(f"policy {args.policy}: "
              f"{sum(p_.numel() for p_ in policy.parameters())/1e9:.2f}B params, no adapter")

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
        # The seed must go *through* evaluate(), not just be set before it: evaluate()
        # re-seeds internally (it offsets per rank so ranks draw independent diffusion
        # timesteps), and its default would otherwise overwrite the seed set here — every
        # repeat would then evaluate identically and the reported std would be a
        # structural zero rather than a measurement. Job 2991038 reported
        # std=0.0000 spread=0.0000 over two seeds, which is what surfaced this.
        losses.append(
            T.evaluate(policy, loader, preprocessor, autocast_ctx, device, seed=args.seed + i)
        )
        print(f"  repeat {i} (seed {args.seed + i}): {losses[-1]:.4f}")

    mean = sum(losses) / len(losses)
    spread = (max(losses) - min(losses)) if len(losses) > 1 else 0.0
    std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5 if len(losses) > 1 else 0.0
    print(f"\nVAL_LOSS mean={mean:.4f} std={std:.4f} spread={spread:.4f} "
          f"n={len(losses)} checkpoint={args.adapter or args.policy}")


if __name__ == "__main__":
    main()
