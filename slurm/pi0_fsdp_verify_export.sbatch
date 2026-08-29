#!/bin/bash
#SBATCH -J pi0-verify-export
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-now
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH -t 00:40:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail

# Does the exported full-FT checkpoint actually load and evaluate?
#
# Writing a directory that *looks* like a policy is not the same as writing one that
# `PI0Policy.from_pretrained` can consume: the weights are gathered from FSDP shards, and
# the normalisation statistics live in a separate processor file that is easy to forget.
# This is the check that the artefact is real, run on the fast queue on one GPU.
#
# Pass the exported directory as the first argument.

CKPT=${1:?usage: sbatch slurm/pi0_fsdp_verify_export.sbatch <model-dir-or-repo> [val_episodes] [repeats]}
# Default 4 to match a smoke run's split. Pass 40 to compare against the production runs —
# a 4-episode val set is ~490 frames and far too noisy to judge small differences on.
VAL_EPISODES=${2:-4}
REPEATS=${3:-2}

if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi

export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}

source "$SCRATCH/vla_finetuning/venvs/vla/bin/activate"
mkdir -p logs

echo "verifying $CKPT"
# May be a local directory or a Hub repo id (e.g. lerobot/pi0 as a control), so listing it
# is best-effort — under `set -e` a bare `ls` on a repo id would abort the job.
ls -la "$CKPT" 2>/dev/null || echo "(not a local path; treating as a Hub repo id)"

# The split must match the run that produced the checkpoint, or "validation" means a
# different set of episodes and the number is not comparable to anything.
python scripts/eval_adapter.py --policy "$CKPT" --val-episodes "$VAL_EPISODES" \
  --batch-size 8 --num-workers 6 --repeats "$REPEATS"
