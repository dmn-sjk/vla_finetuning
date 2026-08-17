"""Check that the pieces which have historically broken on this cluster still work.

Each of these has failed here at least once, and every one of them fails at the
*first training step* rather than at import, which is an expensive place to find out:

  - torch resolving to a CPU build (login node has no GPU driver, so
    `--torch-backend=auto` silently picks CPU)
  - torchvision from PyPI against a torch from the CUDA index ("operator
    torchvision::nms does not exist")
  - torchcodec unable to find FFmpeg 7 shared libraries (Athena ships none;
    they come from a micromamba prefix via LD_LIBRARY_PATH)

Run on a compute node — the GPU checks need a real device:

    srun -p plgrid-now -A <account> --gres=gpu:1 -t 00:10:00 \
        python scripts/verify_env.py
"""

import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.runtime import preload_npp  # noqa: E402  (must precede torchcodec import)

FAILURES: list[str] = []


def check(name: str):
    """Decorator: run a check, record failure, keep going.

    Deliberately does not stop at the first failure — knowing that three things
    are broken in one run beats discovering them one submission at a time.
    """
    def wrap(fn):
        print(f"\n=== {name}")
        try:
            fn()
            print(f"    OK")
        except Exception:
            FAILURES.append(name)
            print("    FAIL")
            traceback.print_exc(limit=3)
        return fn
    return wrap


@check("torch + CUDA")
def _torch():
    import torch
    print(f"    torch {torch.__version__}")
    assert "+cu" in torch.__version__, (
        f"CPU-only build: {torch.__version__}. Reinstall from the CUDA index."
    )
    assert torch.cuda.is_available(), "CUDA not available (are you on a compute node?)"
    print(f"    device: {torch.cuda.get_device_name(0)}")
    print(f"    bf16 supported: {torch.cuda.is_bf16_supported()}")
    x = torch.randn(64, 64, device="cuda")
    assert torch.isfinite(x @ x).all(), "matmul produced non-finite values"


@check("torchvision C++ ops")
def _tv():
    import torch, torchvision
    from torchvision.ops import nms
    print(f"    torchvision {torchvision.__version__}")
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.0, 1.0]])
    # Touches the compiled extension; this is what fails when torchvision and
    # torch come from different indexes.
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).numel() >= 1


@check("torchcodec + FFmpeg shared libs")
def _tc():
    print(f"    NPP preloaded: {preload_npp()}")
    import torchcodec
    from torchcodec.decoders import VideoDecoder
    print(f"    torchcodec {torchcodec.__version__}")

    # Import alone is not proof: torchcodec probes several FFmpeg majors and only
    # fails when a decode is actually attempted. Encode a throwaway clip with the
    # same FFmpeg build we link against, then decode it.
    ffmpeg = Path(os.environ.get("SCRATCH", "")) / "vla_finetuning/ffmpeg/bin/ffmpeg"
    if not ffmpeg.exists():
        print("    (no micromamba ffmpeg binary; skipped round-trip decode)")
        return
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        subprocess.run(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10:duration=1",
             "-pix_fmt", "yuv420p", str(clip)],
            check=True,
        )
        frames = VideoDecoder(str(clip))[0:5]
        print(f"    decoded {tuple(frames.shape)} {frames.dtype}")
        assert frames.shape[0] == 5, f"expected 5 frames, got {frames.shape[0]}"


@check("lerobot pi0 policy")
def _lerobot():
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: F401
    import importlib.metadata as md

    # CODEBASE_VERSION lives in lerobot_dataset on 0.4.x and dataset_metadata on 0.6.x.
    try:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    except ImportError:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    print(f"    lerobot {md.version('lerobot')}  CODEBASE_VERSION={CODEBASE_VERSION}")
    print(f"    transformers {md.version('transformers')} (stock — 0.4.x needed a fork)")
    # LIBERO ports are tagged v3.0 only; on a v2.1 lerobot they are unreachable.
    assert CODEBASE_VERSION.startswith("v3"), (
        f"{CODEBASE_VERSION} cannot fetch the v3.0-tagged LIBERO datasets"
    )


@check("peft LoRA")
def _peft():
    import peft
    from peft import LoraConfig, get_peft_model
    import importlib.metadata as md
    print(f"    peft {md.version('peft')}")
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"])


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}  exe={sys.executable}")
    if FAILURES:
        print(f"\nFAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("\nAll checks passed.")
