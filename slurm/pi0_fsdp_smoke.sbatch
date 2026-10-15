#!/bin/bash
#SBATCH -J pi0-fsdp-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
# Deliberately modest: this is a correctness test, not a throughput measurement, and a
# 4-GPU job that also demands every idle core on a partially-allocated node waits for a
# whole free node instead of slotting into the gap on a busy one.
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Smoke test for the *full fine-tuning* path (train_pi0_fsdp.py), the counterpart of
# slurm/pi0_smoke.sbatch for the LoRA+DDP path. Five phases, each one a way this has
# actually broken:
#
#   1. FULL_SHARD trains at all               — the custom auto-wrap policy fires
#   2. resume at the same world size          — sharded checkpoint round-trips
#   3. SHARD_GRAD_OP + full-model export      — the publishable artefact is written
#   4. LoRA under FSDP                        — mixed requires_grad inside a unit
#   5. resume at a *different* world size     — DCP reshards 4-rank shards onto 2
#
# Phase 5 runs LoRA rather than full FT for a measured reason: **full FT does not fit on
# two 40 GB cards.** Job 2989671 resharded the checkpoint onto 2 ranks correctly and then
# OOM'd in the first step at 35.4 GiB allocated, needing 2.94 GiB more than the card has.
# The resharding path is what phase 5 tests, and LoRA exercises the same DCP code with a
# configuration that fits; the world-size floor for full FT is a benchmark row, not a
# smoke-test assertion.
#
# There is no single-GPU phase: accelerate only engages FSDP from a distributed launch,
# and full FT on one 40 GB card OOMs by design — that measurement lives in
# slurm/pi0_fsdp_bench.sbatch.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502

OUT=$VLA_RUNS/pi0_fsdp_smoke
rm -rf "$OUT" "$OUT-zero2" "$OUT-lora"
COMMON="--dataset lerobot/libero_spatial_image --batch-size 2 --num-workers 6 \
        --val-episodes 4 --grad-checkpointing --lr-warmup-steps 2"

launch () {  # launch <nproc> <args...>
  local n=$1; shift
  srun torchrun \
    --nnodes="$SLURM_NNODES" --nproc_per_node="$n" \
    --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    train_pi0_fsdp.py "$@"
}

echo "=============== phase 1/5: 4 GPUs, full FT, FULL_SHARD ==================="
launch 4 $COMMON --out "$OUT" --finetune full --sharding full_shard \
  --epochs 1 --max-steps 4 --no-export

echo "=============== phase 2/5: 4 GPUs, resume ==============================="
launch 4 $COMMON --out "$OUT" --finetune full --sharding full_shard \
  --epochs 2 --max-steps 8 --no-export --resume

echo "=============== phase 3/5: 4 GPUs, SHARD_GRAD_OP + export ==============="
launch 4 $COMMON --out "$OUT-zero2" --finetune full --sharding shard_grad_op \
  --epochs 1 --max-steps 4

echo "=============== phase 4/5: 4 GPUs, LoRA under FSDP ======================"
launch 4 $COMMON --out "$OUT-lora" --finetune lora --lora-target attention \
  --sharding full_shard --epochs 1 --max-steps 4 --no-export

echo "=============== phase 5/5: 2 GPUs, resume across world size ============="
launch 2 $COMMON --out "$OUT-lora" --finetune lora --lora-target attention \
  --sharding full_shard --epochs 2 --max-steps 8 --no-export --resume

echo "=============== smoke test complete ====================================="
echo "exported full-FT checkpoint:"
ls -la "$OUT-zero2/model_best" 2>/dev/null || echo "  MISSING — export did not run"
