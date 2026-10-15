# slurm/common.sh — shared prologue for every job script in slurm/, and for
# interactive use. Sourced, never executed:
#
#   source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
#
# Nothing in here is specific to one user: every path comes from a variable with
# a default, so another site overrides what differs and touches no job script.
# `sbatch` propagates the submitting environment, so exporting a variable in your
# shell (or in ~/.bashrc) is enough to change every job:
#
#   VLA_ROOT           work root — venv, caches, run outputs [default: ./work inside
#                      the repo, so a fresh clone runs with nothing exported]
#   VLA_VENV           virtualenv to activate [default: first of
#                      $UV_PROJECT_ENVIRONMENT, $VIRTUAL_ENV, $VLA_ROOT/venvs/vla, ./.venv]
#   VLA_RUNS           checkpoints and run outputs    [default: $VLA_ROOT/runs]
#   VLA_FFMPEG_PREFIX  prefix holding FFmpeg 7 shared libs [default: $VLA_ROOT/ffmpeg]
#
# HF_HOME, TRITON_CACHE_DIR, WANDB_DIR, LIBERO_CONFIG_PATH and MUJOCO_GL are also
# only defaulted, so an already-exported value wins.
#
# The SLURM account and partition are not set here — `#SBATCH` directives cannot
# expand variables — but SLURM reads them from the environment, and an
# environment value overrides the directive in the script:
#
#   export SBATCH_ACCOUNT=<account>
#   export SBATCH_PARTITION=<gpu-partition>

VLA_REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$VLA_REPO" || return 1

# Secrets (HF_TOKEN, WANDB_API_KEY) from the gitignored .env, if present.
# python-dotenv is not installed, so the shell must source it — Python won't.
# HF_TOKEN is required: pi0's tokenizer comes from the gated google/paligemma-3b-pt-224.
if [ -f "$VLA_REPO/.env" ]; then
  set -a; . "$VLA_REPO/.env"; set +a
fi

# Self-contained by default: ./work inside the repo, gitignored. Export VLA_ROOT to
# put the venv, the caches and the run outputs on another filesystem instead.
export VLA_ROOT="${VLA_ROOT:-$VLA_REPO/work}"
export VLA_RUNS="${VLA_RUNS:-$VLA_ROOT/runs}"
VLA_FFMPEG_PREFIX="${VLA_FFMPEG_PREFIX:-$VLA_ROOT/ffmpeg}"

# One directory holds every cache, rather than scattering them through $HOME.
export HF_HOME="${HF_HOME:-$VLA_ROOT/hf}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLA_ROOT/.triton}"
export WANDB_DIR="${WANDB_DIR:-$VLA_ROOT/wandb}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_ROOT/libero_config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1
mkdir -p "$VLA_RUNS" "$WANDB_DIR" logs results

# torchcodec needs FFmpeg 7 shared libs. Where the system has none they come from a
# micromamba prefix under $VLA_ROOT (see README); NPP is handled in-process by
# src/runtime.py. A system that ships FFmpeg 7 needs neither.
if [ -d "$VLA_FFMPEG_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$VLA_FFMPEG_PREFIX/lib:${LD_LIBRARY_PATH:-}"
elif ! ldconfig -p 2>/dev/null | grep -q 'libavutil\.so\.59'; then
  echo "common.sh: no FFmpeg 7 libs at $VLA_FFMPEG_PREFIX/lib and none system-wide;" >&2
  echo "           'import torchcodec' will fail. See README -> FFmpeg." >&2
fi

if [ -z "${VLA_VENV:-}" ]; then
  for _cand in "${UV_PROJECT_ENVIRONMENT:-}" "${VIRTUAL_ENV:-}" \
               "$VLA_ROOT/venvs/vla" "$VLA_REPO/.venv"; do
    if [ -n "$_cand" ] && [ -x "$_cand/bin/python" ]; then VLA_VENV="$_cand"; break; fi
  done
  unset _cand
fi
if [ -z "${VLA_VENV:-}" ] || [ ! -x "$VLA_VENV/bin/python" ]; then
  echo "common.sh: no virtualenv found${VLA_VENV:+ at $VLA_VENV}. Create one with" >&2
  echo "    export UV_PROJECT_ENVIRONMENT=$VLA_ROOT/venvs/vla" >&2
  echo "    uv sync" >&2
  echo "or point VLA_VENV at an existing one." >&2
  # Inside a job this must be fatal even in the scripts that run without `set -e`
  # (the OOM sweeps); interactively, `exit` would close the user's shell.
  if [ -n "${SLURM_JOB_ID:-}" ]; then exit 1; else return 1; fi
fi
export VLA_VENV
if [ "${VIRTUAL_ENV:-}" != "$VLA_VENV" ]; then
  . "$VLA_VENV/bin/activate"
fi
