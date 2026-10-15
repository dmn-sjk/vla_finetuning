#!/bin/bash
#SBATCH -J pi0-full
#SBATCH --nodes=1
# Four GPUs, not eight, for two measured reasons. (a) Four is the floor: full FT OOMs on
# 1 and on 2 (job 2990654), so this is the smallest allocation that runs, and on a
# partition where most nodes are drained a 4-GPU job schedules while an 8-GPU one waits
# for a whole free node. (b) Throughput is 19.2 samples/s, so 52.5k samples x 5 epochs is
# ~3.8 h of training. On 8 GPUs, halve --grad-accum to keep the effective batch at 256.
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
# 3.8 h training + 5 validation passes + 45 GB checkpoint writes per epoch, with margin.
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Full fine-tuning of all 4.03B pi0 parameters on LIBERO-spatial, sharded with FSDP.
#
# This is NOT the production recipe — slurm/pi0_train.sbatch (LoRA) is. Full FT of a 4B
# model on 392 episodes is the configuration that invites catastrophic forgetting of
# pi0's pretraining, and its artefact is a multi-GB checkpoint rather than an 11 MB
# adapter. It runs to convergence so the comparison against LoRA is an actual number
# rather than an assumption.
#
# FULL_SHARD rather than SHARD_GRAD_OP: same peak memory (25.1 GiB) and slightly faster
# (19.2 vs 17.5 samples/s) in job 2990654.

source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ib0}

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29504

GPUS_PER_NODE=4
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

# grad-accum 8 puts the effective batch at 8 x 8 x 4 = 256: pi0's own training batch, and
# the same effective batch as the LoRA production run, so the two val curves are
# comparable rather than merely adjacent. That is ~205 optimizer steps per epoch, which is
# why the LR warmup is 50 steps and not the 200-step default.
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train_pi0_fsdp.py \
    --dataset lerobot/libero_spatial_image \
    --finetune full \
    --sharding full_shard \
    --epochs 5 \
    --batch-size 8 \
    --grad-accum 8 \
    --lr 2.5e-5 \
    --lr-warmup-steps 50 \
    --num-workers "$WORKERS" \
    --grad-checkpointing \
    --resume \
    --wandb
