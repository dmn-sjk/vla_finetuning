# Distributed Fine-Tuning of a Vision-Language-Action Model

End-to-end distributed training pipeline for fine-tuning a Vision-Language-Action (VLA) model
for robotic manipulation, built for a multi-GPU SLURM cluster. The model maps multi-camera
visual input and a natural-language instruction to continuous robot control actions.

The project is deliberately split into two phases:

- **Phase 1 — distributed systems foundation.** A native `torch.distributed` training loop:
  NCCL process groups, `DistributedSampler` sharding, rank-gated I/O, gradient accumulation
  with sync control, bf16, gradient checkpointing, and resumable checkpointing. *Implemented
  and verified.*
- **Phase 2 — multimodal scaling.** LoRA fine-tuning of **π0** (~3B: a PaliGemma backbone
  plus a flow-matching action expert) on the **LIBERO** manipulation benchmark via
  `lerobot`, `accelerate`, and `peft`, sharded with FSDP / DeepSpeed ZeRO. *In progress.*

π0 is used in place of the OpenVLA-7B / SmolVLA named in the original brief. It is the
architecture the field converged on and is a first-class `lerobot` policy, so it reuses the
dataset pipeline directly; SmolVLA at 450M is too small for sharding to be load-bearing, and
OpenVLA-7B is not a `lerobot` policy and would need a separate RLDS data path. LIBERO is the
benchmark π0 and OpenVLA both report against, which keeps our numbers comparable.

`train.py` runs identically at world size 1 and under `torchrun`, so the distributed code path
is developed on a single GPU and scaled without changes.

Developed on **Athena** (Cyfronet / PLGrid): 8× NVIDIA A100-SXM4-**40 GB** per node,
InfiniBand, SLURM. The 40 GB card is the binding constraint and is why LoRA is required rather
than optional — see `CLAUDE.md` for the full engineering notes.

## Layout

```
train.py              # Phase 1 entrypoint (world size 1 or N, unchanged)
train_pi0.py          # Phase 2 entrypoint: pi0 + LoRA on LIBERO, same DDP skeleton
src/dist_utils.py     # process group setup, rank gating, cross-rank reductions
src/toy.py            # VLA-shaped synthetic data + small model (Phase 1)
src/runtime.py        # native-library preloading (NPP) needed before torchcodec
scripts/verify_env.py # checks the pieces that have historically broken here
scripts/probe_pi0.py  # one pi0+LoRA forward/backward: batch keys, VRAM, param counts
slurm/train.sbatch    # Phase 1: 8-GPU production job
slurm/smoke.sbatch    # Phase 1: 1-GPU + 4-GPU + resume smoke test
slurm/pi0_train.sbatch# Phase 2: 8-GPU production job
slurm/pi0_smoke.sbatch# Phase 2: 1-GPU + resume + 4-GPU smoke test
pyproject.toml        # dependencies + CUDA wheel index pins
uv.lock               # exact pinned versions
```

Phase 1's `train.py` is kept as its own deliverable rather than being overwritten — the
native `torch.distributed` loop is the point of that phase. Phase 2 builds beside it and
reuses `src/dist_utils.py`.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/vla_finetuning/venvs/vla
uv sync
```

`UV_PROJECT_ENVIRONMENT` keeps the virtualenv off the (small) home quota. On a machine without
that constraint you can omit it and use the default `.venv/`.

The CUDA wheel index is pinned to **cu128** in `pyproject.toml`, and `torch`, `torchvision`
and `torchcodec` must all come from that same index — a PyPI `torchvision` against an index
`torch` fails at import with `operator torchvision::nms does not exist`. Do not install torch
with `--torch-backend=auto` on a machine whose GPU driver is not visible (e.g. a cluster login
node) — it silently resolves to the CPU build.

After setup, verify the pieces that actually break here:

```bash
srun -p plgrid-gpu-a100 -A <account> --gres=gpu:1 -t 00:10:00 \
  python scripts/verify_env.py
```

### lerobot version is load-bearing

Two independent constraints pin **lerobot 0.6.x** (hence Python 3.12):

1. `lerobot` fetches hub datasets by **git tag matching its own `CODEBASE_VERSION`**, not
   from `main`. lerobot 0.3.x is `v2.1`; the official LIBERO ports carry only a `v3.0` tag,
   so they are *silently* unreachable from 0.3.x.
2. lerobot 0.4.x's π0 refuses to build against stock transformers — it imports
   `transformers.models.siglip.check`, which exists only on the HF branch
   `fix/lerobot_openpi`. That branch has since been rebased onto transformers 5.x, which
   violates 0.4.x's own `transformers<5` cap, so **0.4.x + π0 is unsatisfiable today**.
   0.5+ dropped the fork and runs π0 on stock transformers.

lerobot 0.6.1's `torch<2.12` / `torchcodec<0.12` caps are what put the stack on cu128.

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

torchcodec also links **NVIDIA NPP** (`libnppicc.so.12`) for GPU colour conversion, which
torch does not ship — it has no use for it. The `nvidia-npp-cu12` wheel is therefore a
direct dependency, and `src/runtime.py:preload_npp()` loads it `RTLD_GLOBAL` before
torchcodec is imported. That keeps the fix inside the process rather than requiring another
`LD_LIBRARY_PATH` entry, so **entrypoints must call `preload_npp()` before importing
`lerobot` or `torchcodec`**.

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
| — | — | **You must also accept the licence for [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224)** with the same account. π0's tokenizer is loaded from that gated repo, so without it training dies at processor construction with a 403. The π0 weights themselves (`lerobot/pi0`) are *not* gated, which makes this easy to miss. |
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
