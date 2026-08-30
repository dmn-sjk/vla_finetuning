#!/bin/bash
#SBATCH -J pi0-libero-diag
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH -t 00:50:00
#SBATCH -o logs/%x-%j.out
set -uo pipefail

# The LoRA adapter scores 0.3344 validation loss and 0/15 closed-loop success. Exactly one
# of those is wrong about the policy, and this job says which: an open-loop check that uses
# no simulator at all, then a state/image match that anchors our env observation to a real
# dataset frame. See scripts/diagnose_libero.py.

cd "$SLURM_SUBMIT_DIR"
[ -f .env ] && { set -a; . ./.env; set +a; }

export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}
export LIBERO_CONFIG_PATH=$SCRATCH/vla_finetuning/libero_config
export MUJOCO_GL=egl
export PYTHONUNBUFFERED=1

source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
mkdir -p logs

CKPT_ROOT=${CKPT_ROOT:-$PLG_GROUPS_STORAGE/plgguseby/dmns/vla}

echo "################ LoRA adapter_best ################"
python scripts/diagnose_libero.py \
  --policy lerobot/pi0 --adapter "$CKPT_ROOT/pi0_libero/adapter_best" --task-id 0

echo
echo "################ base pi0 (control: open-loop should be clearly worse) ################"
python scripts/diagnose_libero.py --policy lerobot/pi0 --task-id 0 2>&1 | sed -n '/1. OPEN LOOP/,/beats the mean/p'
