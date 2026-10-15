"""Where this project's files live, resolved identically in Python and in shell.

`slurm/common.sh` exports VLA_ROOT, VLA_RUNS and VLA_FFMPEG_PREFIX for every job,
so a job and an interactive run write to the same places. The fallbacks below
repeat that script's defaults, so an entrypoint launched with nothing exported
still lands somewhere sensible instead of in the repository.
"""

import os
from pathlib import Path


_REPO = Path(__file__).resolve().parent.parent


def vla_root() -> Path:
    """Work root: venv, caches and run outputs.

    Defaults to `work/` inside the repository, matching common.sh, so a clone with
    nothing exported stays self-contained. Export VLA_ROOT to move it elsewhere.
    """
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
