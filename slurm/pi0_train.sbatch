#!/bin/bash
#SBATCH -J pi0-libero
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
# Measured throughput is ~85 samples/s on 8 GPUs, so 10 epochs over LIBERO-spatial's
# ~52.5k train samples is roughly 2 hours. Requesting 8h leaves generous margin while
# scheduling far sooner than a 2-day ask. --resume makes a requeue cheap anyway.
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Secrets (HF_TOKEN, WANDB_API_KEY) from the gitignored .env, if present.
# python-dotenv is not installed, so the shell must source it — Python won't.
# HF_TOKEN is required: pi0's tokenizer comes from the gated google/paligemma-3b-pt-224.
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

# Caches off $HOME — the 10 GB quota fills fast, and SLURM refuses jobs past 90%.
export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export WANDB_DIR=$SCRATCH/vla_finetuning/wandb

# eth0/eth1 are DOWN on Athena compute nodes; only ib* is up. Without this NCCL
# can select a dead interface and hang until timeout — the classic "job holds
# 8 GPUs and does nothing" failure. Do NOT set NCCL_IB_DISABLE=1: IB is the fast path.
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO             # drop to WARN once the run is stable

# torchcodec needs FFmpeg shared libs; Athena has none, so they come from a
# micromamba prefix on $SCRATCH. NPP is handled in-process by src/runtime.py.
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}

# Measured on one A100-40GB (see CLAUDE.md): at per-device batch 8, peak VRAM is
# 9.8 GiB with gradient checkpointing and 21.6 GiB without. Batch 16 + checkpointing
# therefore sits around 12 GiB, well inside 40 GB, and gives an effective batch of
# 16 x 2 x 8 = 256 — the batch size the pi0 paper trains at.
#
# Match dataloader workers to allocated cores, leaving headroom for the main proc.
# LIBERO stores frames as PNGs, so decode is CPU-bound and workers matter more here
# than they did in Phase 1 — this, not VRAM, is the likely bottleneck.
GPUS_PER_NODE=8
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
mkdir -p logs

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_pi0.py \
    --dataset lerobot/libero_spatial_image \
    --epochs 7 \
    --batch-size 16 \
    --grad-accum 2 \
    --lora-target attention \
    --lora-r 32 \
    --num-workers "$WORKERS" \
    --grad-checkpointing \
    --resume \
    --wandb
