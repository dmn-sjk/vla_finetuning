#!/bin/bash
#SBATCH -J pi0-libero-verify
# No -A: SLURM picks your default account. Override without editing this file:
#   export SBATCH_ACCOUNT=<your-grant>-gpu-a100   (an env value beats a #SBATCH line)
#SBATCH -p plgrid-now
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# Preflight for the LIBERO success-rate evaluation. Two questions, both of which would
# otherwise be answered by a silently wrong number in the final table:
#
#   1. Does the simulator render at all on a compute node? MuJoCo needs a GL backend and
#      Athena has no X server; libEGL_nvidia.so.0 is present, osmesa is not.
#   2. Are the frames we hand the policy oriented the way its training data was? robosuite
#      renders origin-bottom-left and the RLDS->LeRobot port flips. Feed the policy
#      upside-down images and it runs fine and scores ~0%, with nothing in the logs to say
#      why. --verify-obs settles it against real dataset frames.
#
# Then a 4-episode rollout on the *base* policy to prove the loop closes and to measure
# seconds/episode, which is what sizes the full run.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

echo "############ 1. renderer + camera orientation ############"
python scripts/eval_libero.py --verify-obs

echo
echo "############ 2. 4-episode smoke rollout, base pi0 ############"
time python scripts/eval_libero.py \
  --policy lerobot/pi0 \
  --task-ids 0 1 \
  --episodes-per-task 2 \
  --envs-per-rank 2 \
  --out results/smoke_base.json

echo "############ done ############"
