#!/bin/bash
#SBATCH -J vla-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o logs/smoke-%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517
OUT=$VLA_RUNS/smoke

echo "########## 1 GPU (no torchrun) ##########"
rm -rf $OUT
python train.py --epochs 2 --batch-size 32 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT

echo "########## resume on 1 GPU ##########"
python train.py --epochs 3 --batch-size 32 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT --resume

echo "########## 4 GPUs via torchrun, grad-accum 2 ##########"
rm -rf $OUT
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --epochs 2 --batch-size 8 --grad-accum 2 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT
echo "########## ALL DONE rc=$? ##########"
