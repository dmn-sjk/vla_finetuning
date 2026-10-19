# Shared preamble for every slurm/*.sbatch and for interactive use. Source it, do not run it:
#   source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"
# Each variable below is only defaulted, so an exported value wins:
#   VLA_ROOT (./work), VLA_RUNS ($VLA_ROOT/runs), VLA_FFMPEG_PREFIX ($VLA_ROOT/ffmpeg),
#   VLA_VENV (first of $UV_PROJECT_ENVIRONMENT, $VIRTUAL_ENV, $VLA_ROOT/venvs/vla, ./.venv),
#   NCCL_SOCKET_IFNAME (ib0), HF_HOME, TRITON_CACHE_DIR, WANDB_DIR, LIBERO_CONFIG_PATH.
# Set the SLURM account and partition with SBATCH_ACCOUNT / SBATCH_PARTITION.

VLA_REPO="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$VLA_REPO" || return 1

# secrets from the gitignored .env; HF_TOKEN is needed for the gated PaliGemma tokenizer
if [ -f "$VLA_REPO/.env" ]; then
  set -a; . "$VLA_REPO/.env"; set +a
fi

export VLA_ROOT="${VLA_ROOT:-$VLA_REPO/work}"
export VLA_RUNS="${VLA_RUNS:-$VLA_ROOT/runs}"
VLA_FFMPEG_PREFIX="${VLA_FFMPEG_PREFIX:-$VLA_ROOT/ffmpeg}"

# keep every cache under VLA_ROOT, out of $HOME
export HF_HOME="${HF_HOME:-$VLA_ROOT/hf}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$VLA_ROOT/.triton}"
export WANDB_DIR="${WANDB_DIR:-$VLA_ROOT/wandb}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VLA_ROOT/libero_config}"
export PYTHONUNBUFFERED=1

# NCCL can bootstrap over a down interface and hang; pin it. Tensor traffic still uses IB,
# so do not set NCCL_IB_DISABLE. Without InfiniBand, export an interface that is up.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"

mkdir -p "$VLA_RUNS" "$WANDB_DIR" logs results

# torchcodec needs FFmpeg 7 shared libs; see README for the micromamba prefix
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
  # exit in a job (some scripts lack set -e); return interactively to keep the shell
  if [ -n "${SLURM_JOB_ID:-}" ]; then exit 1; else return 1; fi
fi
export VLA_VENV
if [ "${VIRTUAL_ENV:-}" != "$VLA_VENV" ]; then
  . "$VLA_VENV/bin/activate"
fi
