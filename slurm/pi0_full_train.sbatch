#!/bin/bash
#SBATCH -J pi0-full
#SBATCH --nodes=1
# 4 GPUs is the floor for full FT; on 8, halve --grad-accum to keep the effective batch
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Full fine-tuning of pi0 on LIBERO-spatial under FSDP FULL_SHARD (comparison run, not production).

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29504

GPUS_PER_NODE=4
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

# effective batch 8 x 8 x 4 = 256, as the LoRA run; ~205 steps/epoch, hence the short warmup
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_pi0_fsdp.py \
    --dataset lerobot/libero_spatial_image \
    --finetune full \
    --sharding full_shard \
    --epochs 5 \
    --batch-size 8 \
    --grad-accum 8 \
    --lr 2.5e-5 \
    --lr-warmup-steps 50 \
    --num-workers "$WORKERS" \
    --grad-checkpointing \
    --resume \
    --wandb
