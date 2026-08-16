#!/bin/bash
#SBATCH -J vla-smoke
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 00:25:00
#SBATCH -o logs/smoke-%j.out
set -euo pipefail
# Secrets (HF_TOKEN, WANDB_API_KEY) from the gitignored .env, if present.
# python-dotenv is not installed, so the shell must source it — Python won't.
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

export HF_HOME=$SCRATCH/vla_finetuning/hf
export WANDB_DIR=$SCRATCH/vla_finetuning/wandb
export NCCL_SOCKET_IFNAME=ib0

# torchcodec (lerobot video decode) needs FFmpeg 7 shared libs; Athena has none,
# so they come from a micromamba prefix on $SCRATCH. See CLAUDE.md.
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29517
source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
OUT=$SCRATCH/vla_finetuning/runs/smoke

echo "########## 1 GPU (no torchrun) ##########"
rm -rf $OUT
python train.py --epochs 2 --batch-size 32 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT

echo "########## resume on 1 GPU ##########"
python train.py --epochs 3 --batch-size 32 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT --resume

echo "########## 4 GPUs via torchrun, grad-accum 2 ##########"
rm -rf $OUT
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  train.py --epochs 2 --batch-size 8 --grad-accum 2 --num-workers 4 \
  --train-samples 2048 --val-samples 512 --grad-checkpointing --out $OUT
echo "########## ALL DONE rc=$? ##########"
