"""One forward+backward of pi0+LoRA on LIBERO, to pin down the API and the memory cost.

Everything this prints is something the real trainer needs to assume: the batch keys
after preprocessing, the trainable-parameter fraction under LoRA, and peak VRAM for a
given batch size. Running this first is cheaper than discovering a wrong assumption
inside an 8-GPU job.

    srun -p plgrid-now -A <account> --gres=gpu:1 --cpus-per-task=16 --mem=100G \
        -t 00:30:00 python scripts/probe_pi0.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import preload_npp  # noqa: E402

preload_npp()  # must precede any torchcodec import (lerobot pulls it in transitively)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from lerobot.datasets.factory import resolve_delta_timestamps  # noqa: E402
from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.policies.pi0.configuration_pi0 import PI0Config  # noqa: E402

REPO_ID = "lerobot/libero_spatial_image"
BASE = "lerobot/pi0"
BATCH_SIZE = 2


def main():
    device = torch.device("cuda")

    meta = LeRobotDatasetMetadata(REPO_ID)
    print(f"dataset {REPO_ID}: {meta.total_episodes} episodes, "
          f"{meta.total_frames} frames, fps={meta.fps}")
    print(f"tasks: {list(meta.tasks.index)[:3]} ...")

    cfg = PI0Config(pretrained_path=BASE, device="cuda", dtype="bfloat16")

    # pi0 predicts a chunk of future actions, so the dataset must hand back an action
    # window rather than a single step. lerobot derives the window from the config's
    # delta indices; building it by hand is the classic way to get a silent shape bug.
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    print(f"delta_timestamps keys: {list((delta_timestamps or {}).keys())}")

    ds = LeRobotDataset(REPO_ID, delta_timestamps=delta_timestamps)
    policy = make_policy(cfg, ds_meta=meta)

    total_before = sum(p.numel() for p in policy.parameters())
    policy = policy.wrap_with_peft()
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"params: base={total_before/1e9:.2f}B total={total/1e9:.2f}B "
          f"trainable={trainable/1e6:.2f}M ({100*trainable/total:.3f}%)")

    preprocessor, _ = make_pre_post_processors(
        cfg, pretrained_path=BASE,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": meta.stats,
                "features": {**cfg.input_features, **cfg.output_features},
                "norm_map": cfg.normalization_mapping,
            },
        },
    )

    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    raw = next(iter(loader))
    print("\nraw batch:")
    for k, v in sorted(raw.items()):
        print(f"  {k:36s} {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")

    batch = preprocessor(raw)
    print("\npreprocessed batch:")
    for k, v in sorted(batch.items()):
        if torch.is_tensor(v):
            print(f"  {k:36s} {tuple(v.shape)} {v.dtype} {v.device}")

    policy.to(device).train()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, out = policy.forward(batch)
    loss.backward()
    print(f"\nloss={loss.item():.4f}  extra_keys={list(out)[:5]}")
    print(f"peak VRAM at batch_size={BATCH_SIZE}: "
          f"{torch.cuda.max_memory_allocated()/2**30:.2f} GiB")

    grads = sum(1 for p in policy.parameters() if p.requires_grad and p.grad is not None)
    print(f"params with grads: {grads} (should equal the LoRA adapter count)")


if __name__ == "__main__":
    main()
