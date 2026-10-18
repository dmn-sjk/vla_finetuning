#!/bin/bash
#SBATCH -J pi0-fsdp-bench
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail    # not -e: some rows are expected to OOM

# Sharding sweep: 30 steps per row, one THROUGHPUT or OOM_EXPECTED line each.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29503

OUT=$VLA_RUNS/pi0_fsdp_bench
rm -rf "$OUT"
# short val pass and no multi-GiB writes; neither falls inside the timed window
COMMON="--dataset lerobot/libero_spatial_image --num-workers 12 --val-episodes 4 \
        --max-steps 30 --epochs 1 --warmup-steps 5 --lr-warmup-steps 2 \
        --no-export --no-checkpoint --out $OUT"

# rows needing more GPUs than allocated are skipped, so `sbatch --gres=gpu:4` gives a partial sweep
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
  # 20 min cap per row: an OOM'd rank can leave the others waiting on the NCCL timeout
  timeout 1200 srun torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$n" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_pi0_fsdp.py "$@" \
    || echo "ROW_FAILED $label (see traceback above; OOM here is a result, not a bug)"
}

# --- the baseline: no sharding -------------------------------------------------------
# expected to OOM; --oom-ok reports it as a result line
row "1 GPU  | DDP        | full FT" 1 $COMMON --finetune full --sharding ddp \
    --batch-size 8 --grad-checkpointing --oom-ok
# expected to OOM too (DDP replicates optimizer state); no --oom-ok above world size 1
row "8 GPUs | DDP        | full FT" 8 $COMMON --finetune full --sharding ddp \
    --batch-size 8 --grad-checkpointing

# --- the sharded rows ----------------------------------------------------------------
row "8 GPUs | SHARD_GRAD_OP | full FT" 8 $COMMON --finetune full --sharding shard_grad_op \
    --batch-size 8 --grad-checkpointing
row "8 GPUs | FULL_SHARD    | full FT" 8 $COMMON --finetune full --sharding full_shard \
    --batch-size 8 --grad-checkpointing
# without activation checkpointing, to size the activations
row "8 GPUs | FULL_SHARD    | full FT | no grad ckpt" 8 $COMMON --finetune full \
    --sharding full_shard --batch-size 8

# --- how far down the world size can go ----------------------------------------------
row "4 GPUs | SHARD_GRAD_OP | full FT" 4 $COMMON --finetune full --sharding shard_grad_op \
    --batch-size 8 --grad-checkpointing
row "4 GPUs | FULL_SHARD    | full FT" 4 $COMMON --finetune full --sharding full_shard \
    --batch-size 8 --grad-checkpointing
# expected to OOM: locates the world-size floor
row "2 GPUs | FULL_SHARD    | full FT (expect OOM)" 2 $COMMON --finetune full \
    --sharding full_shard --batch-size 8 --grad-checkpointing

# --- the comparison that motivates keeping LoRA in production ------------------------
row "8 GPUs | FULL_SHARD    | LoRA" 8 $COMMON --finetune lora --lora-target attention \
    --sharding full_shard --batch-size 8 --grad-checkpointing

echo "=================== summary ==================="
# find the log by job id: $SLURM_JOB_NAME can differ from -J, and a failed grep fails the job
grep -hE "^(THROUGHPUT|OOM_EXPECTED|ROW_FAILED|ROW_SKIPPED)" \
  "$SLURM_SUBMIT_DIR"/logs/*-"$SLURM_JOB_ID".out || echo "(no result lines found)"
