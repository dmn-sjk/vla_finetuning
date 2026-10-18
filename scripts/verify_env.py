"""Check the training and LIBERO eval environment on a GPU node: `python -m scripts.verify_env`."""

import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from src.paths import ffmpeg_prefix


def check_torch():
    import torch

    assert torch.cuda.is_available(), f"no CUDA: CPU build ({torch.__version__}) or no GPU here"
    return f"torch {torch.__version__}, python {sys.version.split()[0]} at {sys.prefix}"


def check_torchvision():
    import torch
    import torchvision.ops

    # nms fails when torchvision and torch come from different indexes
    torchvision.ops.nms(torch.rand(2, 4, device="cuda"), torch.rand(2, device="cuda"), 0.5)
    return f"torchvision {torchvision.__version__}"


def check_torchcodec():
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    # import alone does not prove FFmpeg loads; decode a generated clip
    ffmpeg = shutil.which("ffmpeg", path=ffmpeg_prefix() / "bin") or shutil.which("ffmpeg")
    if ffmpeg:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as clip:
            cmd = [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=64x64:r=10:d=1"]
            subprocess.run([*cmd, "-pix_fmt", "yuv420p", clip.name], check=True)
            assert VideoDecoder(clip.name)[0:5].shape[0] == 5
    return f"torchcodec {torchcodec.__version__}" + ("" if ffmpeg else " (no ffmpeg, no decode)")


def check_libraries():
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: F401
    from peft import LoraConfig  # noqa: F401

    assert CODEBASE_VERSION.startswith("v3"), f"{CODEBASE_VERSION}: LIBERO ports are v3.0 only"
    return ", ".join(f"{p} {md.version(p)}" for p in ("lerobot", "transformers", "peft"))


def check_cv2():
    # import first: uninstalling the GUI build deletes cv2 but leaves headless's metadata
    import cv2  # noqa: F401

    owners = md.packages_distributions().get("cv2") or []
    assert "opencv-python" not in owners, "GUI opencv-python owns cv2; see README, Simulator"
    return ", ".join(f"{p} {md.version(p)}" for p in sorted(owners))


def check_libero():
    if importlib.util.find_spec("libero") is None:
        return None
    assert "LIBERO_CONFIG_PATH" in os.environ, "LIBERO_CONFIG_PATH is not set; see README"
    cfg = Path(os.environ["LIBERO_CONFIG_PATH"], "config.yaml")
    assert cfg.is_file(), f"no {cfg}: importing libero would block on input(); see README"
    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero import benchmark

    assert benchmark.get_benchmark_dict()["libero_spatial"]().n_tasks == 10
    # resolve assets as get_assets_path() does, without calling it: it downloads into $HOME
    assets = Path(benchmark.__file__).parents[1] / "assets"
    assets = assets if os.path.lexists(assets) else Path.home() / ".cache/libero/assets"
    expected = ("articulated_objects", "stable_scanned_objects", "turbosquid_objects",
                "stable_hope_objects", "scenes")  # fmt: skip
    missing = [d for d in expected if not (assets / d).is_dir()]
    assert not missing, f"{assets} is missing {', '.join(missing)}"
    return f"hf-libero {md.version('hf-libero')}, assets at {assets.resolve()}"


CHECKS = [  # (name, fn, optional): an optional check returning None is ABSENT, not FAIL
    ("torch + CUDA", check_torch, False),
    ("torchvision ops", check_torchvision, False),
    ("torchcodec + FFmpeg", check_torchcodec, False),
    ("lerobot, transformers, peft", check_libraries, False),
    ("headless cv2", check_cv2, False),
    ("LIBERO sim stack (rollouts only)", check_libero, True),
]
if __name__ == "__main__":
    failed = []
    for name, fn, optional in CHECKS:
        try:
            info = fn()
            if optional and info is None:
                print(f"ABSENT  {name}")
            else:
                print(f"OK      {name}: {info}")
        except Exception:
            failed.append(name)
            print(f"FAIL    {name}\n{traceback.format_exc(limit=3)}")
    sys.exit(f"\nFAILED: {', '.join(failed)}" if failed else 0)
