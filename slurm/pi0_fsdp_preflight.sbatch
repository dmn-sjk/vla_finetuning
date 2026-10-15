#!/bin/bash
#SBATCH -J pi0-fsdp-pre
# No -A: SLURM picks your default account. Override without editing this file:
#   export SBATCH_ACCOUNT=<your-grant>-gpu-a100   (an env value beats a #SBATCH line)
#SBATCH -p plgrid-now
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# Everything in train_pi0_fsdp.py that does *not* need more than one GPU, run on the
# fast queue so the 8-GPU allocation is only ever spent on the sharded path itself:
# fp32 model construction, the LR schedule, checkpoint save/resume, the export, and the
# single-GPU OOM measurement that motivates the whole file.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

OUT=$VLA_RUNS/pi0_fsdp_pre
rm -rf "$OUT" "$OUT-lora"
COMMON="--dataset lerobot/libero_spatial_image --num-workers 6 --val-episodes 4 \
        --lr-warmup-steps 2 --grad-checkpointing"

echo "=========== phase 1/4: 1 GPU, LoRA, DDP path, save + export ==========="
python train_pi0_fsdp.py $COMMON --finetune lora --sharding ddp --batch-size 2 \
  --epochs 1 --max-steps 4 --out "$OUT-lora"

echo "=========== phase 2/4: 1 GPU, LoRA, resume ============================"
python train_pi0_fsdp.py $COMMON --finetune lora --sharding ddp --batch-size 2 \
  --epochs 2 --max-steps 8 --resume --out "$OUT-lora"

echo "=========== phase 3/4: the guard against a silently unsharded run ====="
# --sharding full_shard on one process must fail loudly rather than train unsharded.
python train_pi0_fsdp.py $COMMON --finetune full --sharding full_shard --batch-size 2 \
  --epochs 1 --max-steps 1 --no-export --out "$OUT" \
  && echo "GUARD_FAILED: single-process FSDP was accepted" \
  || echo "guard ok: single-process FSDP refused"

echo "=========== phase 4/4: 1 GPU, full FT, unsharded -> expected OOM ======"
python train_pi0_fsdp.py $COMMON --finetune full --sharding ddp --batch-size 8 \
  --epochs 1 --max-steps 20 --no-export --oom-ok --out "$OUT"

echo "=========== preflight complete ========================================"
