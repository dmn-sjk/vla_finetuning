#!/bin/bash
#SBATCH -J pi0-libero-eval
#SBATCH --nodes=1
# 4 GPUs: inference only, and 4-GPU jobs schedule sooner
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# LIBERO-spatial success rates of base pi0, LoRA and full FT (10 tasks x 50 trials each).

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29506

# export CKPT_ROOT to evaluate an archived copy
CKPT_ROOT=${CKPT_ROOT:-$VLA_RUNS}
NGPU=4
NSTEPS=${NSTEPS:-5}
EPT=${EPT:-50}

run () {  # name, extra args...
  local name=$1; shift
  echo "================ $name ================"
  local t0=$SECONDS
  srun torchrun --nnodes=1 --nproc_per_node=$NGPU \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    scripts/eval_libero.py \
      --suite libero_spatial \
      --episodes-per-task "$EPT" \
      --envs-per-rank 8 \
      --n-action-steps "$NSTEPS" \
      --out "results/libero_spatial_${name}.json" \
      "$@"
  echo "---- $name finished rc=$? in $((SECONDS - t0))s ----"
}

# sanity gate: ~0% here means the obs/action pipeline is broken
echo "================ sanity: lora, 2 tasks x 4 trials ================"
srun torchrun --nnodes=1 --nproc_per_node=$NGPU \
  --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  scripts/eval_libero.py --suite libero_spatial \
    --policy lerobot/pi0 --adapter "$CKPT_ROOT/pi0_libero/adapter_best" \
    --task-ids 0 1 --episodes-per-task 4 --envs-per-rank 2 \
    --n-action-steps "$NSTEPS" --out results/sanity_lora.json

run base       --policy lerobot/pi0
run lora_best  --policy lerobot/pi0 --adapter "$CKPT_ROOT/pi0_libero/adapter_best"
run full_best  --policy "$CKPT_ROOT/pi0_full/model_best"

echo
echo "================ summary ================"
grep -h "^SUCCESS_RATE" logs/*-"$SLURM_JOB_ID".out || echo "(no SUCCESS_RATE lines)"
