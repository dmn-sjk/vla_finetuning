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

# LoRA vs full fine-tuning, measured the same way on the same episodes.
#
# The two numbers already in the logs are NOT comparable: they come from different eval
# batch sizes, and the LoRA figure predates the `--repeats` seeding fix in
# scripts/eval_adapter.py. This job re-measures both artefacts with one script, one batch
# size and one set of seeds, which is the only way the comparison means anything.
#
# Base pi0 is included as the reference: without it, "0.33" is a number with no scale.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

COMMON="--val-episodes 40 --batch-size 8 --num-workers 6 --repeats 3"

echo "=================== 1/2: LoRA adapter_best ==================="
python scripts/eval_adapter.py \
  --adapter "$VLA_RUNS/pi0_libero/adapter_best" $COMMON

echo "=================== 2/2: full FT model_best ==================="
python scripts/eval_adapter.py \
  --policy "$VLA_RUNS/pi0_full/model_best" $COMMON

echo "=================== comparison ==================="
echo "base pi0 reference (job 2991110): VAL_LOSS mean=0.8719 std=0.0048 n=3"
grep -hE "^VAL_LOSS" "$SLURM_SUBMIT_DIR"/logs/*-"$SLURM_JOB_ID".out || true
