#!/bin/bash
#SBATCH -J pi0-scaling
# No -A: SLURM picks your default account. Override without editing this file:
#   export SBATCH_ACCOUNT=<your-grant>-gpu-a100   (an env value beats a #SBATCH line)
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Does adding GPUs actually buy wall clock here?
#
# pi0+LoRA fits on one 40 GB card (~10 GiB), so multi-GPU is a *throughput* choice,
# not a capacity one. LIBERO ships PNG frames, so dataloading is CPU-bound decode over
# ~106k small files on Lustre — which is exactly the shape of workload that scales
# sublinearly. This measures it instead of assuming.
#
# EFFECTIVE BATCH IS HELD CONSTANT at 128 by moving grad_accum inversely to world
# size, so every configuration does identical math on identical data and only the
# parallelism changes. Same total samples per config => elapsed time is comparable.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502

OUT=$VLA_RUNS/pi0_scaling
BS=16
STEPS=20          # optimizer steps, after 5 warmup steps that are excluded from timing
EFFECTIVE=128     # held constant across every configuration below

# Workers per rank scale with world size so each rank keeps a full share of the CPUs.
# Starving the input pipeline would make this measure the dataloader, not the GPUs.
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
