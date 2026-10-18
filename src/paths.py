"""Project paths, resolved with the same defaults as slurm/common.sh.

Honours VLA_ROOT, VLA_RUNS and VLA_FFMPEG_PREFIX when exported.
"""

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def vla_root() -> Path:
    """Work root for venv, caches and run outputs; defaults to `work/` in the repo."""
    env = os.environ.get("VLA_ROOT")
    return Path(env) if env else _REPO / "work"


def run_dir(name: str) -> Path:
    """Default --out for one run, e.g. run_dir("pi0_libero")."""
    runs = os.environ.get("VLA_RUNS")
    return (Path(runs) if runs else vla_root() / "runs") / name


def ffmpeg_prefix() -> Path:
    """Prefix holding the FFmpeg 7 shared libraries torchcodec links against."""
    env = os.environ.get("VLA_FFMPEG_PREFIX")
    return Path(env) if env else vla_root() / "ffmpeg"
