#!/bin/bash
#SBATCH -J pi0-libero-sanity
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# Cheap read on the corrected observation pipeline, while the 1500-episode job waits for
# nodes. The earlier smoke (job 3108456) ran before --verify-obs had settled the camera
# orientation, so it used flipud and told us nothing about policy quality -- 0/4 was the
# expected result of feeding a mis-rotated frame, not evidence about the checkpoint.
#
# This runs the fine-tuned adapter on 3 tasks x 5 trials with rot180. If the pipeline is
# right, the adapter should be clearly above zero; if it comes back ~0% the big job is
# going to burn 90 minutes producing three confident zeros and should be cancelled.
# Base pi0 on the same 15 episodes gives the zero-shot floor to read that against.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

# Finished checkpoints are read from where training left them; export CKPT_ROOT to
# read an archived copy instead (see README -> Where things are configured).
CKPT_ROOT=${CKPT_ROOT:-$VLA_RUNS}
ARGS="--suite libero_spatial --task-ids 0 4 8 --episodes-per-task 5 --envs-per-rank 5 --n-action-steps 5"

echo "############ LoRA adapter_best (rot180) ############"
time python scripts/eval_libero.py $ARGS \
  --policy lerobot/pi0 --adapter "$CKPT_ROOT/pi0_libero/adapter_best" \
  --out results/sanity_lora.json

echo
echo "############ base pi0, same 15 episodes ############"
time python scripts/eval_libero.py $ARGS \
  --policy lerobot/pi0 \
  --out results/sanity_base.json

echo "############ done ############"
grep -h "^SUCCESS_RATE" logs/pi0-libero-sanity-"$SLURM_JOB_ID".out || true
