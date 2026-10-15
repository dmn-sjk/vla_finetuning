#!/bin/bash
#SBATCH -J vla-phase1
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

# On an InfiniBand cluster the ethernet interfaces are often down while only ib* is up.
# Without this NCCL can select a dead interface and hang until timeout — the classic
# "job holds 8 GPUs and does nothing" failure. Export NCCL_SOCKET_IFNAME if your fast
# interface is named differently. Do NOT set NCCL_IB_DISABLE=1: IB is the fast path.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export NCCL_DEBUG=INFO             # drop to WARN once the run is stable

# Match dataloader workers to allocated cores, leaving headroom for the main proc.
GPUS_PER_NODE=8
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
    --epochs 10 \
    --batch-size 32 \
    --grad-accum 2 \
    --num-workers "$WORKERS" \
    --grad-checkpointing \
    --resume \
    --wandb
