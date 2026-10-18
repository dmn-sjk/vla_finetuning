#!/bin/bash
#SBATCH -J pi0-scaling
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# LoRA DDP throughput at 1/2/4 GPUs with the effective batch held at 128.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502

OUT=$VLA_RUNS/pi0_scaling
BS=16
STEPS=20          # optimizer steps, after 5 warmup steps that are excluded from timing
EFFECTIVE=128     # held constant across every configuration below

# workers per rank scale with world size so the dataloader is not the bottleneck
for N in 1 2 4; do
  ACCUM=$(( EFFECTIVE / (BS * N) ))
  WORKERS=$(( SLURM_CPUS_PER_TASK / N - 2 ))
  echo "=================== world_size=$N grad_accum=$ACCUM workers=$WORKERS ==================="
  rm -rf "$OUT"
  srun torchrun \
    --nnodes=1 --nproc_per_node="$N" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_pi0.py \
      --dataset lerobot/libero_spatial_image \
      --epochs 1 --max-steps $STEPS --warmup-steps 5 \
      --batch-size $BS --grad-accum "$ACCUM" \
      --num-workers "$WORKERS" --val-episodes 2 \
      --grad-checkpointing --log-every 1000 --out "$OUT"
done

echo "=================== summary ==================="
grep -h "^THROUGHPUT" "logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" || true
