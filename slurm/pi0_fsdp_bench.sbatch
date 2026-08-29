#!/bin/bash
#SBATCH -J pi0-fsdp-bench
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail    # deliberately NOT -e: several rows are *expected* to OOM

# The sharding experiment. Each row runs a few dozen optimizer steps of pi0 full
# fine-tuning and prints one machine-greppable THROUGHPUT (or OOM_EXPECTED) line;
# convergence is not the point, memory and throughput are.
#
# The question it answers: LoRA fits pi0 in 11.7 GiB on a 40 GB card, so what exactly
# does sharding buy? Answer: it is the difference between "does not run at all" and
# "runs", once the 4.03B parameters are all trainable and AdamW needs fp32 moments for
# every one of them (~64 GiB before activations).

if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export WANDB_DIR=$SCRATCH/vla_finetuning/wandb
export NCCL_SOCKET_IFNAME=ib0
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29503

source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
mkdir -p logs

OUT=$SCRATCH/vla_finetuning/runs/pi0_fsdp_bench
rm -rf "$OUT"
# --val-episodes 4 keeps the end-of-run validation pass short; --no-export and
# --no-checkpoint skip writing multi-GiB files nobody will load. None of the three touch
# the measured window, which covers training steps only.
COMMON="--dataset lerobot/libero_spatial_image --num-workers 12 --val-episodes 4 \
        --max-steps 30 --epochs 1 --warmup-steps 5 --lr-warmup-steps 2 \
        --no-export --no-checkpoint --out $OUT"

# Rows that need more GPUs than this allocation has are skipped rather than failing, so
# the same script gives a partial sweep on a 4-GPU allocation when no full node is free —
# which, on a partition where most nodes are drained, is the difference between numbers
# today and numbers tomorrow. Submit with `sbatch --gres=gpu:4 ...` to override.
NGPU=${SLURM_GPUS_ON_NODE:-8}
echo "allocation has $NGPU GPUs"

row () {  # row <label> <nproc> <args...>
  local label=$1 n=$2; shift 2
  if [ "$n" -gt "$NGPU" ]; then
    echo "ROW_SKIPPED $label (needs $n GPUs, allocation has $NGPU)"
    return
  fi
  echo "=================== $label ==================="
  rm -rf "$OUT"
  # Hard 20-minute cap per row: a rank that dies of OOM mid-collective can leave the
  # others waiting out the 30-minute NCCL timeout, and one such row would eat the whole
  # sweep's wall clock.
  timeout 1200 srun torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$n" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_pi0_fsdp.py "$@" \
    || echo "ROW_FAILED $label (see traceback above; OOM here is a result, not a bug)"
}

# --- the baseline: no sharding -------------------------------------------------------
# Expected to OOM. --oom-ok turns that into a reported line instead of a failed job;
# it is only safe at world size 1, where no other rank is left waiting on a collective.
row "1 GPU  | DDP        | full FT" 1 $COMMON --finetune full --sharding ddp \
    --batch-size 8 --grad-checkpointing --oom-ok
# DDP replicates optimizer state on every rank, so 8 GPUs does not help. No --oom-ok:
# if it dies, torchrun tears the whole group down and the row is reported as FAILED.
row "8 GPUs | DDP        | full FT" 8 $COMMON --finetune full --sharding ddp \
    --batch-size 8 --grad-checkpointing

# --- the sharded rows ----------------------------------------------------------------
row "8 GPUs | SHARD_GRAD_OP | full FT" 8 $COMMON --finetune full --sharding shard_grad_op \
    --batch-size 8 --grad-checkpointing
row "8 GPUs | FULL_SHARD    | full FT" 8 $COMMON --finetune full --sharding full_shard \
    --batch-size 8 --grad-checkpointing
# No activation recomputation: isolates how much of the peak is activations rather than
# states, which is what decides whether the batch size can go up.
row "8 GPUs | FULL_SHARD    | full FT | no grad ckpt" 8 $COMMON --finetune full \
    --sharding full_shard --batch-size 8

# --- how far down the world size can go ----------------------------------------------
row "4 GPUs | SHARD_GRAD_OP | full FT" 4 $COMMON --finetune full --sharding shard_grad_op \
    --batch-size 8 --grad-checkpointing
row "4 GPUs | FULL_SHARD    | full FT" 4 $COMMON --finetune full --sharding full_shard \
    --batch-size 8 --grad-checkpointing
# Expected to fail, and that is the point: it locates the world-size floor. Measured in
# job 2989671 at per-device batch 2 — the checkpoint resharded onto 2 ranks correctly, then
# the first step OOM'd at 35.4 GiB allocated, 2.94 GiB short. Four GPUs is the minimum.
row "2 GPUs | FULL_SHARD    | full FT (expect OOM)" 2 $COMMON --finetune full \
    --sharding full_shard --batch-size 8 --grad-checkpointing

# --- the comparison that motivates keeping LoRA in production ------------------------
row "8 GPUs | FULL_SHARD    | LoRA" 8 $COMMON --finetune lora --lora-target attention \
    --sharding full_shard --batch-size 8 --grad-checkpointing

echo "=================== summary ==================="
# Own log, found by job id: $SLURM_JOB_NAME is not always what -J set it to, and a grep
# that cannot open its file exits 2, which marks the whole job FAILED even when every row
# behaved exactly as designed.
grep -hE "^(THROUGHPUT|OOM_EXPECTED|ROW_FAILED|ROW_SKIPPED)" \
  "$SLURM_SUBMIT_DIR"/logs/*-"$SLURM_JOB_ID".out || echo "(no result lines found)"
