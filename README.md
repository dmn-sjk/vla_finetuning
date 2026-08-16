# Distributed Fine-Tuning of a Vision-Language-Action Model

End-to-end distributed training pipeline for fine-tuning a Vision-Language-Action (VLA) model
for robotic manipulation, built for a multi-GPU SLURM cluster. The model maps multi-camera
visual input and a natural-language instruction to continuous robot control actions.

The project is deliberately split into two phases:

- **Phase 1 — distributed systems foundation.** A native `torch.distributed` training loop:
  NCCL process groups, `DistributedSampler` sharding, rank-gated I/O, gradient accumulation
  with sync control, bf16, gradient checkpointing, and resumable checkpointing. *Implemented
  and verified.*
- **Phase 2 — multimodal scaling.** LoRA fine-tuning of a 7B VLA backbone (OpenVLA / SmolVLA)
  on teleoperated trajectory datasets via `lerobot`, `accelerate`, and `peft`, sharded with
  FSDP / DeepSpeed ZeRO. *Not yet started.*

`train.py` runs identically at world size 1 and under `torchrun`, so the distributed code path
is developed on a single GPU and scaled without changes.

Developed on **Athena** (Cyfronet / PLGrid): 8× NVIDIA A100-SXM4-**40 GB** per node,
InfiniBand, SLURM. The 40 GB card is the binding constraint and is why LoRA is required rather
than optional — see `CLAUDE.md` for the full engineering notes.

## Layout

```
train.py              # training entrypoint (world size 1 or N, unchanged)
src/dist_utils.py     # process group setup, rank gating, cross-rank reductions
src/toy.py            # VLA-shaped synthetic data + small model (Phase 2 replaces these)
slurm/train.sbatch    # 8-GPU production job
slurm/smoke.sbatch    # 1-GPU + 4-GPU + resume smoke test
pyproject.toml        # dependencies + CUDA wheel index pins
uv.lock               # exact pinned versions
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/vla_finetuning/venvs/vla
uv sync
```

`UV_PROJECT_ENVIRONMENT` keeps the virtualenv off the (small) home quota. On a machine without
that constraint you can omit it and use the default `.venv/`.

The CUDA wheel index is pinned to **cu130** in `pyproject.toml` because `torchcodec` wheels
link CUDA 13; `torch`, `torchvision`, and `torchcodec` must all come from that same index.
Do not install torch with `--torch-backend=auto` on a machine whose GPU driver is not visible
(e.g. a cluster login node) — it silently resolves to the CPU build.

### FFmpeg

`lerobot` decodes trajectory video through `torchcodec`, which needs FFmpeg **shared
libraries**. These are C libraries, not Python wheels, so `uv` cannot provide them — and the
FFmpeg bundled inside PyAV is unusable here, because `auditwheel` hash-mangles its SONAMEs
specifically to prevent other packages from linking against it. torchcodec's own
documentation recommends conda-forge; `micromamba` is a fast single-binary client for it.
Athena has no FFmpeg module and no system libav, so it is installed into the project
directory:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
./bin/micromamba -r $SCRATCH/.mamba create -y -p $SCRATCH/vla_finetuning/ffmpeg \
  -c conda-forge "ffmpeg=7.*"
```

Jobs put it on the library path (the SLURM scripts already do this):

```bash
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}
```

Without it, `import torchcodec` fails and dataset loading dies at the first video frame. If
your system already provides FFmpeg 7 shared libraries, this step is unnecessary.

### Secrets: `.env` (required)

Training reads credentials from a `.env` file in the repository root. **It is gitignored and
must never be committed.** Create it yourself:

```bash
cat > .env <<'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 .env
```

| Variable | Required | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | yes | Pull gated model weights and push the trained LoRA adapter. Needs **write** scope — a read token fails only at the final push. Create at <https://huggingface.co/settings/tokens>. |
| `WANDB_API_KEY` | optional | Training telemetry. Can be omitted if `~/.netrc` already holds W&B credentials, or if you run without `--wandb`. |

`python-dotenv` is not a dependency, so nothing loads `.env` implicitly — the shell sources
it. `slurm/*.sbatch` do this already:

```bash
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi
```

Running outside SLURM, source it the same way before launching. As an alternative to `.env`,
`hf auth login` stores the token under `$HF_HOME` instead, outside the repository.

## Running

Smoke test — single GPU, 4 GPUs via `torchrun`, and checkpoint resume, in about two minutes:

```bash
sbatch slurm/smoke.sbatch
```

Full run on 8 GPUs:

```bash
sbatch slurm/train.sbatch
```

Iterating locally on one GPU, without SLURM:

```bash
$SCRATCH/vla_finetuning/venvs/vla/bin/python train.py \
  --epochs 3 --batch-size 32 --grad-checkpointing
```

Startup logs the effective batch size (per-device × gradient accumulation × world size), which
is the number that makes runs comparable across different cluster shapes.
