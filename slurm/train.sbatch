#!/bin/bash
#SBATCH -J vla-phase1
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1        # one task per node; torchrun spawns the ranks
#SBATCH --cpus-per-task=128
#SBATCH --mem=480G
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Secrets (HF_TOKEN, WANDB_API_KEY) from the gitignored .env, if present.
# python-dotenv is not installed, so the shell must source it — Python won't.
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

# Caches off $HOME — the 10 GB quota fills fast, and SLURM refuses jobs past 90%.
export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export WANDB_DIR=$SCRATCH/vla_finetuning/wandb

# eth0/eth1 are DOWN on Athena compute nodes; only ib* is up. Without this NCCL
# can select a dead interface and hang until timeout — the classic "job holds
# 8 GPUs and does nothing" failure. Do NOT set NCCL_IB_DISABLE=1: IB is the fast path.
export NCCL_SOCKET_IFNAME=ib0

# torchcodec (lerobot video decode) needs FFmpeg 7 shared libs; Athena has none,
# so they come from a micromamba prefix on $SCRATCH. See CLAUDE.md.
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}
export NCCL_DEBUG=INFO             # drop to WARN once the run is stable

# Match dataloader workers to allocated cores, leaving headroom for the main proc.
GPUS_PER_NODE=8
WORKERS=$(( SLURM_CPUS_PER_TASK / GPUS_PER_NODE - 2 ))

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500

source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
mkdir -p logs

srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node=$GPUS_PER_NODE \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  train.py \
    --epochs 10 \
    --batch-size 32 \
    --grad-accum 2 \
    --num-workers "$WORKERS" \
    --grad-checkpointing \
    --resume \
    --wandb
