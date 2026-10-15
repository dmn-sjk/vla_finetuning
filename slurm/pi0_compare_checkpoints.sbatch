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

# Base pi0 vs LoRA vs full fine-tuning, measured the same way on the same episodes.
#
# The numbers in the training logs are NOT comparable: they come from different eval
# batch sizes, and the LoRA figure predates the `--repeats` seeding fix in
# scripts/eval_adapter.py. This job re-measures all three checkpoints with one script, one
# batch size and one set of seeds, which is the only way the comparison means anything.
#
# Base pi0 is measured as the reference: without it, "0.33" is a number with no scale.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

# Finished checkpoints are read from where training left them; export CKPT_ROOT to
# evaluate an archived copy instead (same convention as pi0_libero_eval.sbatch).
CKPT_ROOT=${CKPT_ROOT:-$VLA_RUNS}

COMMON="--val-episodes 40 --batch-size 8 --num-workers 6 --repeats 3"

# Collected separately: grep cannot read this job's own log while SLURM is writing it.
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
