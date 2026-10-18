#!/bin/bash
#SBATCH -J pi0-libero
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Production run: LoRA fine-tuning of pi0 on LIBERO-spatial, 8 GPUs, DDP.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export NCCL_DEBUG=INFO             # drop to WARN once the run is stable

# effective batch 16 x 2 x 8 = 256; PNG decode is CPU-bound, so match workers to the allocated cores
GPUS_PER_NODE=8
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

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
