#!/bin/bash
#SBATCH -J pi0-libero-eval
#SBATCH --nodes=1
# Four GPUs, not eight: rollouts are inference-only (~10 GiB/rank), so the binding
# constraint is scheduling latency, not memory, and 4-GPU jobs start while 8-GPU ones wait
# for a whole free node. Ranks are independent -- no collective except the final gather.
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# Closed-loop LIBERO-spatial success rates for every checkpoint worth reporting, under one
# protocol so the three numbers are comparable to each other as well as to published
# tables: 10 tasks x 50 trials from the suite's own saved initial states, 280-step cap.
#
# Base pi0 is included deliberately. Published pi0 LIBERO numbers come from a pi0 that was
# fine-tuned on LIBERO; the released `lerobot/pi0` checkpoint was not, so its success rate
# here is the honest zero-shot floor our fine-tuning has to beat, and quoting the paper's
# number as "the baseline" would be wrong.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29506

# Finished checkpoints are read from where training left them; export CKPT_ROOT to
# read an archived copy instead (see README -> Where things are configured).
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

# Sanity gate before ~90 minutes of GPU time. The fine-tuned adapter should clear the floor
# comfortably on two tasks; ~0% here means the observation/action pipeline is broken and
# every number after it would be a confident zero. This is not hypothetical: the first
# attempt at this evaluation scored 0/15 on a checkpoint that is demonstrably good, because
# the post-processor was never given the dataset's stats and handed the simulator z-scores
# (job 3108532). `_assert_unnormalizes` in eval_libero.py now catches that specific fault at
# load time, but this gate is what catches the ones nobody has thought of yet -- it exercises
# state, camera orientation, prompt and action scale together, which no assertion does.
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
