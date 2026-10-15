#!/bin/bash
#SBATCH -J pi0-libero-recheck
# No -A: SLURM picks your default account. Override without editing this file:
#   export SBATCH_ACCOUNT=<your-grant>-gpu-a100   (an env value beats a #SBATCH line)
#SBATCH -p plgrid-now
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 00:55:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# Re-check after the unnormalization fix (job 3108532 proved the post-processor was handing
# the simulator z-scores instead of actions). Two gates, cheapest first:
#   1. open loop -- predicted action chunk vs recorded expert actions, no simulator. Should
#      now beat the "predict the dataset mean" baseline on most dims; before the fix it beat
#      it on 1 of 7.
#   2. closed loop on 15 episodes. A fine-tuned LIBERO-spatial policy should be clearly
#      above zero here.
# Only if both pass is the 1500-episode run worth submitting.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

CKPT_ROOT=${CKPT_ROOT:-${PLG_GROUPS_STORAGE:-}/plgguseby/dmns/vla}
# Not everyone has that group storage: fall back to this user's own run root.
[ -d "$CKPT_ROOT" ] || CKPT_ROOT=$VLA_RUNS
ADAPTER="$CKPT_ROOT/pi0_libero/adapter_best"

echo "################ gate 1: open loop, LoRA adapter_best ################"
python scripts/diagnose_libero.py --policy lerobot/pi0 --adapter "$ADAPTER" --task-id 0 \
  2>&1 | sed -n '/1. OPEN LOOP/,/beats the mean/p'

echo
echo "################ gate 2: closed loop, 3 tasks x 5 trials ################"
time python scripts/eval_libero.py --suite libero_spatial \
  --task-ids 0 4 8 --episodes-per-task 5 --envs-per-rank 5 --n-action-steps 5 \
  --policy lerobot/pi0 --adapter "$ADAPTER" --out results/recheck_lora.json

echo "################ done ################"
