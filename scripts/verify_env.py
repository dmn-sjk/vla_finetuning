"""Check that the pieces which have historically broken on this cluster still work.

Each of these has failed here at least once, and every one of them fails at the
*first training step* rather than at import, which is an expensive place to find out:

  - torch resolving to a CPU build (login node has no GPU driver, so
    `--torch-backend=auto` silently picks CPU)
  - torchvision from PyPI against a torch from the CUDA index ("operator
    torchvision::nms does not exist")
  - torchcodec unable to find FFmpeg 7 shared libraries (Athena ships none;
    they come from a micromamba prefix via LD_LIBRARY_PATH)
  - the LIBERO sim stack half-installed: robosuite's GUI `opencv-python` clobbering the
    headless `cv2`, or `libero` hitting its import-time `input()` with no config path

The sim stack is optional - training never touches it - so its check reports ABSENT
rather than failing when it is simply not installed.

Run on a compute node - the GPU checks need a real device:

    srun -p plgrid-now -A <account> --gres=gpu:1 -t 00:10:00 \
        python scripts/verify_env.py
"""

import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import ffmpeg_prefix  # noqa: E402

FAILURES: list[str] = []


class Absent(Exception):
    """An optional component is not installed. Not a failure.

    Only the sim stack is optional: `uv sync` deliberately leaves it out, so a
    training-only environment is *correct* without it. A component that is present
    but broken still fails - a half-installed sim stack is worse than none, because
    it surfaces inside a SLURM allocation instead of here.
    """


def check(name: str):
    """Decorator: run a check, record failure, keep going.

    Deliberately does not stop at the first failure - knowing that three things
    are broken in one run beats discovering them one submission at a time.
    """

    def wrap(fn):
        print(f"\n=== {name}")
        try:
            fn()
            print("    OK")
        except Absent as e:
            print(f"    ABSENT: {e}")
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
    import torch
    import torchvision
    from torchvision.ops import nms

    print(f"    torchvision {torchvision.__version__}")
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 1.0, 1.0]])
    # Touches the compiled extension; this is what fails when torchvision and
    # torch come from different indexes.
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).numel() >= 1


@check("torchcodec + FFmpeg shared libs")
def _tc():
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    print(f"    torchcodec {torchcodec.__version__}")

    # Import alone is not proof: torchcodec probes several FFmpeg majors and only
    # fails when a decode is actually attempted. Encode a throwaway clip with the
    # same FFmpeg build we link against, then decode it.
    ffmpeg = ffmpeg_prefix() / "bin/ffmpeg"
    if not ffmpeg.exists():
        system = shutil.which("ffmpeg")
        if system is None:
            print("    (no ffmpeg binary found; skipped round-trip decode)")
            return
        ffmpeg = Path(system)
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "clip.mp4"
        subprocess.run(
            [
                str(ffmpeg),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=64x64:rate=10:duration=1",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ],
            check=True,
        )
        frames = VideoDecoder(str(clip))[0:5]
        print(f"    decoded {tuple(frames.shape)} {frames.dtype}")
        assert frames.shape[0] == 5, f"expected 5 frames, got {frames.shape[0]}"


@check("lerobot pi0 policy")
def _lerobot():
    import importlib.metadata as md

    from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # noqa: F401

    # CODEBASE_VERSION lives in lerobot_dataset on 0.4.x and dataset_metadata on 0.6.x.
    try:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    except ImportError:
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    print(f"    lerobot {md.version('lerobot')}  CODEBASE_VERSION={CODEBASE_VERSION}")
    print(f"    transformers {md.version('transformers')} (stock - 0.4.x needed a fork)")
    # LIBERO ports are tagged v3.0 only; on a v2.1 lerobot they are unreachable.
    assert CODEBASE_VERSION.startswith("v3"), (
        f"{CODEBASE_VERSION} cannot fetch the v3.0-tagged LIBERO datasets"
    )


@check("peft LoRA")
def _peft():
    import importlib.metadata as md

    from peft import LoraConfig

    print(f"    peft {md.version('peft')}")
    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj"])


@check("headless cv2 (no GUI opencv)")
def _cv2():
    import importlib.metadata as md

    # Not optional: lerobot declares `opencv-python-headless>=4.9,<4.14` as a hard
    # dependency and the training path imports cv2, so a cv2-less env is never correct.
    #
    # Import first, because the metadata lies. `opencv-python` and
    # `opencv-python-headless` ship the *same* `cv2` module under different distribution
    # names, so uninstalling the GUI one deletes the shared `cv2/` files while leaving
    # headless's metadata in place -- ownership then reports a perfectly healthy cv2 that
    # cannot be imported. The sim install passes through exactly that state.
    import cv2

    owners = md.packages_distributions().get("cv2") or []
    print(f"    cv2 {cv2.__version__} provided by: {', '.join(sorted(owners)) or '(unknown)'}")

    # No resolver ever reports the conflict, since the distribution names differ --
    # whichever installed last wins the files. robosuite requires the GUI build, so the sim
    # install drags it in, and a GUI cv2 on a headless node dies at the first render with
    # `libGL.so.1: cannot open shared object file`. Ownership is the test, not presence.
    assert "opencv-python" not in owners, (
        "opencv-python (GUI) owns cv2 and has clobbered the headless build; uninstall it "
        'and reinstall opencv-python-headless (see README, "Simulator")'
    )


@check("libero sim stack (rollouts only)")
def _libero():
    import importlib.metadata as md
    import importlib.util
    import os

    # Safe to ask first: find_spec executes nothing, and the top-level `libero/__init__.py`
    # is empty anyway - the interactive prompt lives one level down in `libero/libero/`.
    if importlib.util.find_spec("libero") is None:
        raise Absent(
            "not installed - training does not need it. Closed-loop rollouts do; "
            'see README, "Simulator, for evaluation only".'
        )

    # libero/libero/__init__.py prompts via input() at import time whenever its config.yaml
    # does not exist, hanging a batch job until stdin EOFs (EOFError when stdin is closed).
    # The trigger is the *file*, not the env var: common.sh exports LIBERO_CONFIG_PATH but
    # never creates the config, so asserting on the variable alone would pass here and then
    # hang on the import below.
    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH")
    assert cfg_dir, "LIBERO_CONFIG_PATH is unset - source slurm/common.sh first"
    assert Path(cfg_dir, "config.yaml").is_file(), (
        f"{cfg_dir}/config.yaml does not exist - importing libero would prompt via input() "
        'and hang this job. Create it first; see README, "Simulator".'
    )

    # robosuite's version comes from metadata rather than `import robosuite`: importing it
    # pulls mujoco's EGL bindings, which die with an opaque OpenGL NoneType error on any
    # host with no libEGL on the library path. Nothing here needs the module itself.
    import libero.libero
    from libero.libero import benchmark, get_libero_path

    print(f"    hf-libero {md.version('hf-libero')}  robosuite {md.version('robosuite')}")
    print(f"    mujoco {md.version('mujoco')}  bddl files: {get_libero_path('bddl_files')}")
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    print(f"    libero_spatial: {suite.n_tasks} tasks")

    # The 409 MB of meshes is the piece most likely to be missing or mis-wired. Resolve it
    # the same way libero does, and NEVER by calling get_assets_path(): that function is not
    # a read-only probe. It returns the package `assets/` dir whenever that path resolves and
    # otherwise *downloads* 409 MB into a hardcoded ~/.cache/libero -- so a check built on it
    # would quietly consume the very $HOME quota it exists to protect, then pass against the
    # fresh copy and hide the breakage. Both wirings are legitimate: a redirected cache
    # (what README sets up) or an explicit symlink inside the package. Order matters --
    # get_assets_path() prefers the package path, so a complete cache cannot rescue an
    # incomplete package dir.
    candidates = [
        Path(libero.libero.__file__).parent / "assets",
        Path.home() / ".cache" / "libero" / "assets",
    ]
    assets = next((c for c in candidates if os.path.lexists(c)), None)
    assert assets is not None, (
        "no asset tree found: neither the package `assets/` path nor ~/.cache/libero/assets "
        "exists, so the first rollout would download 409 MB into $HOME. Redirect the cache "
        'to scratch first; see README, "Simulator".'
    )
    assert assets.is_dir(), f"{assets} is a dangling symlink -> {os.readlink(assets)}"

    # Upstream's own freshness check (download_utils.py) looks at the first four of these;
    # `scenes` is in the published tree and just as load-bearing, so check all five.
    missing = [
        d
        for d in (
            "articulated_objects",
            "stable_scanned_objects",
            "turbosquid_objects",
            "stable_hope_objects",
            "scenes",
        )
        if not (assets / d).is_dir()
    ]
    assert not missing, (
        f"assets at {assets} are incomplete (missing {', '.join(missing)}); rollouts will "
        'fail at the first object lookup. See README, "Simulator".'
    )
    target = Path(os.readlink(assets)) if assets.is_symlink() else assets
    print(f"    assets: {assets}" + (f" -> {target}" if assets.is_symlink() else ""))
    # Informational: a real directory under $HOME is the quota problem, not an error.
    if not target.is_absolute() or str(target.resolve()).startswith(str(Path.home())):
        print(f"    note: assets live under $HOME ({target}) - watch the quota")

    # Imports and task metadata only. This does NOT prove the EGL renderer works - that
    # needs MUJOCO_GL=egl on a compute node, and is exercised by scripts/eval_libero.py.


if __name__ == "__main__":
    print(f"python {sys.version.split()[0]}  exe={sys.executable}")
    if FAILURES:
        print(f"\nFAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("\nAll checks passed.")
