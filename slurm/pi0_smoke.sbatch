#!/bin/bash
#SBATCH -J pi0-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Same three phases as the Phase 1 smoke test: single GPU, resume, then multi-GPU.
# The point is to prove the distributed path before spending 8 GPUs for two days.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29501

OUT=$VLA_RUNS/pi0_smoke
rm -rf "$OUT"
# Mirror the production LoRA target set: which modules carry adapters changes which
# parameters reach the DDP reducer, so smoking with different targets than production
# does not actually test the production path.
COMMON="--dataset lerobot/libero_spatial_image --batch-size 2 --num-workers 6 \
        --val-episodes 4 --grad-checkpointing --lora-target attention --out $OUT"

echo "=================== phase 1/3: single GPU, no torchrun ==================="
python train_pi0.py $COMMON --epochs 1 --max-steps 4

echo "=================== phase 2/3: single GPU, resume ========================"
python train_pi0.py $COMMON --epochs 2 --max-steps 8 --resume

echo "=================== phase 3/3: 4 GPUs via torchrun ======================="
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_pi0.py $COMMON --epochs 3 --max-steps 12 --grad-accum 2 --resume

echo "=================== smoke test complete =================================="
