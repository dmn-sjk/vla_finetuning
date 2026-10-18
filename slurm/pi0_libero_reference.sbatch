#!/bin/bash
#SBATCH -J pi0-libero-ref
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# lerobot's LIBERO-finetuned pi0 through our harness (own config and stats), to locate our gap.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29507

REF=lerobot/pi0_libero_finetuned_v044
COMMON="--suite libero_spatial --dataset HuggingFaceVLA/libero --policy $REF \
        --from-pretrained-config --checkpoint-stats --envs-per-rank 8"

launch () { srun torchrun --nnodes=1 --nproc_per_node=4 \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" -m scripts.eval_libero "$@"; }

echo "================ sanity: 3 tasks x 5 trials ================"
# shellcheck disable=SC2086
launch $COMMON --task-ids 0 4 8 --episodes-per-task 5 --n-action-steps 5 \
  --out results/ref_sanity.json

echo
echo "================ A: 500 episodes, n_action_steps=50 (its own config default) ================"
t0=$SECONDS
# shellcheck disable=SC2086
launch $COMMON --episodes-per-task 50 --n-action-steps 50 \
  --out results/libero_spatial_ref_v044_nas50.json
echo "---- A rc=$? in $((SECONDS-t0))s ----"

echo
echo "================ B: 500 episodes, n_action_steps=5 (matches our 54.2% protocol) ================"
t0=$SECONDS
# shellcheck disable=SC2086
launch $COMMON --episodes-per-task 50 --n-action-steps 5 \
  --out results/libero_spatial_ref_v044_nas5.json
echo "---- B rc=$? in $((SECONDS-t0))s ----"

echo
echo "================ summary ================"
grep -h "^SUCCESS_RATE" logs/pi0-libero-ref-"$SLURM_JOB_ID".out || echo "(none)"
echo "ours for comparison: LoRA 54.2 +- 2.2, full FT 56.6 +- 2.2, base 0.0 (n_action_steps=5)"
