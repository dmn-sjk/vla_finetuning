#!/bin/bash
#SBATCH -J pi0-compare
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 00:55:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Val loss of base pi0, LoRA and full FT under one protocol (same episodes, batch size, seeds).

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

# export CKPT_ROOT to evaluate an archived copy
CKPT_ROOT=${CKPT_ROOT:-$VLA_RUNS}

COMMON="--val-episodes 40 --batch-size 8 --num-workers 6 --repeats 3"

# collected separately: grep cannot read this job's own log while SLURM writes it
RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT

echo "=================== 1/3: LoRA adapter_best ==================="
python scripts/eval_adapter.py \
  --adapter "$CKPT_ROOT/pi0_libero/adapter_best" $COMMON | tee -a "$RESULTS"

echo "=================== 2/3: full FT model_best ==================="
python scripts/eval_adapter.py \
  --policy "$CKPT_ROOT/pi0_full/model_best" $COMMON | tee -a "$RESULTS"

echo "=================== 3/3: base lerobot/pi0 ==================="
python scripts/eval_adapter.py --policy lerobot/pi0 $COMMON | tee -a "$RESULTS"

echo "=================== comparison ==================="
grep -E "^VAL_LOSS" "$RESULTS" || true
