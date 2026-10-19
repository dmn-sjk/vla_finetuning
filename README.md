# Distributed Fine-Tuning of a Vision-Language-Action Model

Code used to fine-tune π0 (`lerobot`'s implementation) on the LIBERO-Spatial manipulation benchmark on a multi-GPU SLURM cluster. It produced two checkpoints on the Hugging Face Hub:

1. LoRA: [pi0-libero-spatial-lora](https://huggingface.co/dmnsjk/pi0-libero-spatial-lora) ([model card](model_cards/pi0-libero-spatial-lora.md))
1. Full fine-tuning under FSDP: [pi0-libero-spatial-full](https://huggingface.co/dmnsjk/pi0-libero-spatial-full) ([model card](model_cards/pi0-libero-spatial-full.md))

## Results

LIBERO-spatial, 40 held-out episodes for val loss (3 seeds), 500 closed-loop episodes (10 tasks x 50 trials) for success rate.

| checkpoint | trainable | val loss (± std) | success (± SE) | artefact | min 40 GB GPUs |
| --- | --- | --- | --- | --- | --- |
| base `lerobot/pi0` | - | 0.8719 ± 0.0048 | 0.0% | - | - |
| LoRA `adapter_best` | 12.65M | 0.3344 ± 0.0020 | 54.2% ± 2.2 | 49 MB | 1 |
| full FT `model_best` | 3.3B | 0.3350 ± 0.0014 | 56.6% ± 2.2 | 7.6 GB | 4 |

LoRA and full FT tie on both metrics. Full FT does not fit on one A100-40GB, nor on 8 under DDP; it needs FSDP across at least 4 GPUs. π0 is 3.3B parameters as published. lerobot's port holds 4.03B because it also keeps two vocabulary-sized LM heads (~0.79B) that π0 never uses.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

1. Everything downloaded or produced (venv, HF and Triton caches, FFmpeg, run outputs) lives under `$VLA_ROOT`, which defaults to `work/` in the repository:

   ```bash
   export VLA_ROOT=/path/to/work
   export UV_PROJECT_ENVIRONMENT=$VLA_ROOT/venvs/vla
   uv sync
   ```

1. FFmpeg 7 shared libraries, for `torchcodec`. Skip if the system provides them. `slurm/common.sh` puts `$VLA_FFMPEG_PREFIX/lib` on `LD_LIBRARY_PATH` otherwise.

   ```bash
   curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
   ./bin/micromamba -r $VLA_ROOT/.mamba create -y -p $VLA_ROOT/ffmpeg \
     -c conda-forge "ffmpeg=7.*"
   ```

1. Credentials in a gitignored `.env` (or log in with the CLIs). Accept the licence for [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) with the same HF account: π0's tokenizer loads from that gated repo.

   ```bash
   cat > .env <<'EOF'
   HF_TOKEN=<hf_token>
   WANDB_API_KEY=<wandb_api_key>
   EOF
   chmod 600 .env
   ```

1. Verify the setup:

   ```bash
   source slurm/common.sh        # venv, caches, .env
   python -m scripts.verify_env
   ```

`slurm/common.sh` only defaults each variable (`VLA_RUNS`, `VLA_VENV`, `VLA_FFMPEG_PREFIX`, the caches, `NCCL_SOCKET_IFNAME=ib0`; full list in its header), so an exported value wins. Evaluation and publishing read checkpoints from `$CKPT_ROOT`, else `$VLA_RUNS`.

### Simulator (evaluation only)

Closed-loop rollouts need LIBERO, installed by hand because `robomimic` depends on `egl-probe`, which compiles against EGL headers and would break `uv sync`. `-c constraints.txt` keeps the install additive. **`uv sync` deletes all of this**; use `uv sync --inexact` afterwards.

```bash
source slurm/common.sh
uv pip install -c constraints.txt --no-deps hf-libero==0.1.4 robomimic==0.2.0
uv pip install -c constraints.txt \
  "hydra-core>=1.2,<1.4" "robosuite==1.4.0" "bddl==1.0.1" easydict einops thop \
  "matplotlib>=3.5.3" "cloudpickle>=2.0.0" "mujoco>=3.0,<3.9" "future>=0.18.2" \
  h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg
uv pip uninstall opencv-python && \
  uv pip install -c constraints.txt --reinstall-package opencv-python-headless \
    "opencv-python-headless==4.12.0.88"
```

Assets, linked into LIBERO's cache, and the config file LIBERO otherwise asks for via `input()` at import time:

```bash
source slurm/common.sh
hf download lerobot/libero-assets --repo-type dataset \
  --local-dir "$VLA_ROOT/libero_assets"
mkdir -p ~/.cache/libero
ln -sfn "$VLA_ROOT/libero_assets" ~/.cache/libero/assets

SITE=$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
assets: $VLA_ROOT/libero_assets
bddl_files: $SITE/libero/libero/bddl_files
benchmark_root: $SITE/libero/libero
datasets: $SITE/libero/libero/../datasets
init_states: $SITE/libero/libero/init_files
EOF
```

## Running

Job scripts source `slurm/common.sh` themselves. Run it yourself before the plain `python` commands. Val loss is stochastic (flow matching), so always evaluate with `--repeats`.

```bash
sbatch slurm/pi0_train.sbatch                 # LoRA, 8 GPUs, DDP
sbatch slurm/pi0_full_train.sbatch            # full FT, FSDP
sbatch slurm/pi0_libero_eval.sbatch           # success rates, base vs LoRA vs full FT
python -m scripts.eval_adapter --adapter $VLA_RUNS/pi0_libero/adapter_best --repeats 5
python -m scripts.eval_adapter --policy $VLA_RUNS/pi0_full/model_best --repeats 5
scripts/publish.sh                            # dry run; --go to upload
```

One GPU, no SLURM: `python train_pi0.py --max-steps 20 --val-episodes 2 --grad-checkpointing --out $VLA_RUNS/pi0_dev`.

## Layout

```
train_pi0.py          # LoRA, DDP
train_pi0_fsdp.py     # full FT under FSDP via accelerate
src/                  # LoRA presets, data split, evaluation, distributed and path helpers
scripts/              # verify_env, eval_adapter, eval_libero, render_results, publishing
slurm/                # common.sh preamble + one .sbatch per experiment
results/              # per-episode rollout records behind the success rates
model_cards/          # Hub READMEs of both checkpoints
checkpoints.sha256    # hashes of the evaluated checkpoints
docs/notes.md         # non-obvious constraints and full result tables
constraints.txt       # pins for installing the simulator into the venv
```

## Limitations and future work

Published π0 reaches ~97% on LIBERO-spatial (openpi). These checkpoints reach 54-57%. Ruled out: LoRA itself (full FT ties it), the eval harness (standard protocol), and weight decay, Adam betas and peak LR (full FT used the reference values). Next levers, in order:

1. Run lerobot's own LIBERO-finetuned π0 through this harness (`slurm/pi0_libero_reference.sbatch`) to split the gap between this recipe and the lerobot port, where users report ~69% ([lerobot#2114](https://github.com/huggingface/lerobot/issues/2114)).
1. Sweep `n_action_steps` (5 is untuned; inference only, no retraining).
1. Train on all four suites (`HuggingFaceVLA/libero`, 1,693 vs 392 episodes; full FT overfits after one epoch on spatial), with effective batch 32-64 instead of 256 (LoRA ran 1,314 steps to openpi's 30,000) and `empty_cameras=1` to match π0's pretrained 3-camera layout.
1. Evaluate out of distribution on the other suites.
1. EMA of weights (openpi uses 0.999).

## Licence

MIT, see [LICENSE](LICENSE).
