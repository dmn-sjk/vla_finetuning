# CLAUDE.md

Distributed fine-tuning of a Vision-Language-Action (VLA) model for robotic manipulation,
on the Athena GPU cluster (Cyfronet / PLGrid). Full spec: `md_files/PROJECT.md`.

Goal: LoRA fine-tune a VLA model (OpenVLA-7B / SmolVLA) on teleoperated robot trajectory
data (LIBERO, ALOHA, PushT), mapping multi-camera frames + language commands to continuous
actions. Deliverables: adapters on HF Hub, a public W&B report, and the SLURM + distributed
training code.

## Current state

**Phase 1 is implemented and verified on the cluster.** Phase 2 (lerobot + Accelerate + PEFT,
real VLA backbone) has not been started.

```
train.py              # Phase 1 loop: DDP, grad accum, bf16, checkpoint/resume, W&B
src/dist_utils.py     # process group setup, rank gating, count-weighted reductions
src/toy.py            # VLA-shaped synthetic data + small model (Phase 2 swaps these out)
slurm/train.sbatch    # 8-GPU production job
slurm/smoke.sbatch    # short 1-GPU + 4-GPU + resume smoke test
pyproject.toml        # dependency declarations + CUDA index pins
uv.lock               # exact pinned versions — commit this
```

`train.py` runs identically at world size 1 (`python train.py`) and under `torchrun`, so
develop on 1 GPU and scale unchanged. Verified on Athena (job 2948190, on the current cu130
stack): 1 GPU, 4 GPUs via torchrun, and resume-from-checkpoint on both. `effective_batch` is
logged on startup.

Phase 2 should keep `train.py`'s structure and replace `src/toy.py` with real lerobot
datasets and the VLA backbone — the batch dict keys (`images`, `state`, `tokens`, `actions`)
are the seam.

Run the smoke test after any change to the distributed path:

```bash
sbatch slurm/smoke.sbatch    # 1 GPU + 4 GPU + resume, ~2 min on plgrid-gpu-a100
```

**Before starting Phase 2**, one thing is still missing: **no Hugging Face token is
configured**, so pushing adapters to the Hub (deliverable #2) will fail. See Secrets below.

Model and dataset choice is still open — `PROJECT.md` lists OpenVLA-7B / SmolVLA and
LIBERO / ALOHA / PushT. Confirm with the user before committing to one.

## Cluster facts (verified on Athena, 2026-08-16)

These are measured, not assumed. Re-verify before relying on them if much time has passed.

| Property | Value |
| --- | --- |
| Login node | `login01.athena.cyfronet.pl` |
| GPUs | **8× NVIDIA A100-SXM4-40GB** per node (not 80GB) |
| CPU / RAM | 128 cores, 1 TB RAM per node |
| Interconnect | InfiniBand, `ib0`–`ib3` up; **`eth0`/`eth1` are DOWN on compute nodes** |
| Grant account | `plgttaautopilot3-gpu-a100` |
| Partitions | `plgrid-gpu-a100` (2 days), `plgrid-now` (1 h, for smoke tests), `8k3-gpu` (14 days) |
| `$SCRATCH` | `/net/tscratch/people/plgdmnsjk` — 12 TB quota |
| Home | `/net/people/plgrid/plgdmnsjk` — **10 GB quota, ~6 GB used; `sbatch` fails past 90%** |
| Node-local disk | None (`TmpDisk=0`); `/tmp` is a 15 GB tmpfs that consumes RAM |
| Internet on compute nodes | **Yes** — HF/W&B reachable from inside jobs |
| Site PyTorch modules | Stale (max 1.13.1) — do not use; `uv sync` instead |
| FFmpeg | **Absent** — no module, no system libav; provided via micromamba (see below) |

### The 40 GB constraint drives everything

A 7B model in bf16 is ~14 GB of weights before activations or optimizer state. On 40 GB cards:

- **LoRA is mandatory, not an optimization.** Full fine-tuning of 7B needs ~112 GB of AdamW
  state alone; even ZeRO-3 across 8 GPUs leaves no headroom.
- Gradient checkpointing + bf16 are the default, not a fallback.
- Prefer FSDP `SHARD_GRAD_OP` or DeepSpeed ZeRO-2 for LoRA (base weights are frozen, so
  sharding them buys little and costs all-gathers). Reach for ZeRO-3 / full shard only if OOM.
- Develop against **SmolVLA (~450M)** for fast iteration, then scale the same code path to
  OpenVLA-7B. Do not debug distributed plumbing at 7B — it wastes grant hours.

### Home quota will break things silently

A 10 GB home means a single HF model download fills it, and past 90% **SLURM refuses to
accept jobs at all**. Only code lives in home; everything that grows lives on `$SCRATCH`,
grouped under one project directory:

```
$SCRATCH/vla_finetuning/
├── hf/            # HF_HOME — models + datasets
├── runs/          # checkpoints (train.py --out default)
├── wandb/         # WANDB_DIR
├── .triton/       # TRITON_CACHE_DIR
├── ffmpeg/        # micromamba prefix, see below
└── venvs/vla/     # UV_PROJECT_ENVIRONMENT
```

`slurm/*.sbatch` export all of these. Outside a job, set them yourself:

```bash
export HF_HOME=$SCRATCH/vla_finetuning/hf
export TRITON_CACHE_DIR=$SCRATCH/vla_finetuning/.triton
export WANDB_DIR=$SCRATCH/vla_finetuning/wandb
```

`$SCRATCH/.uv-cache` deliberately stays outside the project dir — it is a shared package
cache, not project data. Note `$SCRATCH` is *temporary* storage subject to purging; the
contents above are all either re-downloadable or re-creatable from `uv.lock`, but
**checkpoints you care about should be copied elsewhere**.

## Environment

Use `uv` (0.10.2, already installed at `~/.local/bin/uv`). Site PyTorch modules are too old
for modern `lerobot`/`peft`/FSDP; use pip wheels with bundled CUDA and skip the CUDA module
unless something needs `nvcc`.

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock` — **both are committed,
and `uv.lock` is the reproducibility guarantee**. Recreate the environment anywhere with:

```bash
export UV_CACHE_DIR=$SCRATCH/.uv-cache
export UV_PROJECT_ENVIRONMENT=$SCRATCH/vla_finetuning/venvs/vla   # NOT $HOME (10 GB quota)
uv sync
```

`UV_PROJECT_ENVIRONMENT` is essential: `uv sync` otherwise creates `.venv/` in the project
directory, which lives in the 10 GB home. Add or change a dependency with `uv add <pkg>` /
edit `pyproject.toml` then `uv lock`; never hand-install into the venv, or the lock drifts.

Installed and verified: torch 2.13.0+cu130, torchvision 0.28.0+cu130, torchcodec 0.16.0+cu130,
transformers 4.57.6, accelerate 1.14.0, peft 0.20.0, lerobot 0.3.2, wandb 0.28.2, numpy 2.2.6.

Three pins in `pyproject.toml` are load-bearing and are commented there — don't "simplify" them:

- **cu130, not cu128.** torchcodec wheels link CUDA 13; on a cu128 torch it is unimportable.
  The Athena driver (595.71.05) supports CUDA 13 and A100 (sm_80) is fully supported.
- **torch, torchvision and torchcodec all come from the same index.** A PyPI torchvision
  against an index torch fails at import with `operator torchvision::nms does not exist`.
- **`transformers<5`.** peft is not yet compatible with transformers 5.x — importing it
  raises `Could not import module 'BloomPreTrainedModel'`.

Never use `--torch-backend=auto`: it resolves against the *login* node, which has no GPU
driver, and silently installs the CPU build. The explicit index in `pyproject.toml` avoids
this entirely. Also note the venv has no `pip` (`uv venv` omits it) — `pip install` after
activating falls through to the system conda pip and installs into the wrong environment.

Never run training on the login node — it has GPUs in `sinfo` but jobs belong on compute nodes.

### Secrets

Credentials live in a gitignored `.env` at the repo root, `chmod 600`. `README.md` documents
the format for a fresh clone; there is deliberately no `.env.example`. `HF_TOKEN` needs
**write** scope to push adapters. This repo is an open-source deliverable, so a committed
token would be published — the ignore rule matches exactly `.env`, so a `.env.bak` or
`.env.local` would *not* be caught.

A `.env` with a valid write-scoped `HF_TOKEN` is already in place (user `dmnsjk`, verified
against the Hub API). `huggingface_hub.get_token()` picks it up once the shell sources the
file.

`python-dotenv` is **not** installed, so nothing reads `.env` automatically. The shell must
source it, which `slurm/*.sbatch` already do:

```bash
if [ -f "${SLURM_SUBMIT_DIR:-.}/.env" ]; then
  set -a; . "${SLURM_SUBMIT_DIR:-.}/.env"; set +a
fi
```

`hf auth login` is the alternative — it writes the token to `$HF_HOME/token`, which sits on
`$SCRATCH`, outside the repo, so it cannot be committed by accident. Note `HF_HOME` defaults
to `~/.cache/huggingface`, in the 10 GB home; always export it to the project dir first.
W&B credentials are already in `~/.netrc`, so `WANDB_API_KEY` is optional.

### FFmpeg (required for lerobot video decoding)

Athena has **no FFmpeg** — no module, no system libav — and torchcodec needs its shared
libraries. FFmpeg 7 is installed to `$SCRATCH/vla_finetuning/ffmpeg` via micromamba, and jobs
must put it on the library path (both `slurm/*.sbatch` already do):

```bash
export LD_LIBRARY_PATH=$SCRATCH/vla_finetuning/ffmpeg/lib:${LD_LIBRARY_PATH:-}
```

**Why not uv?** FFmpeg is a C library, not a Python wheel — it is not on PyPI. PyAV bundles
FFmpeg but `auditwheel` hash-mangles the SONAMEs (`libavcodec-5e6b81c2.so.62`) precisely so
other packages cannot link them, and it ships FFmpeg 8 while torchcodec wants 7. torchcodec's
README recommends conda-forge and offers no pip route. There is no FFmpeg module on Athena
(verified on compute nodes), and no root for a system package.

`apptainer` and `singularity` *are* available on compute nodes (not on the login node) — a
container is the alternative worth considering if this ever needs to reproduce off PLGrid,
at the cost of a rebuild on every dependency change.

To recreate it (`conda` also works but its solver takes >15 min here; micromamba takes
seconds — and set `-r $SCRATCH/.mamba` so the package cache doesn't fill `$HOME`):

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
./bin/micromamba -r $SCRATCH/.mamba create -y -p $SCRATCH/vla_finetuning/ffmpeg -c conda-forge "ffmpeg=7.*"
```

Without this on the path, `import torchcodec` fails and lerobot dataset loading dies at the
first video frame. There is no working fallback: lerobot's `pyav` path calls
`torchvision.io.VideoReader`, which torchvision 0.28 removed. Worse, lerobot picks its
default backend with `importlib.util.find_spec("torchcodec")`, which only checks the module
*exists* — so a broken torchcodec is selected anyway and fails later at decode time rather
than at import.

## SLURM conventions

Every batch script needs `-A plgttaautopilot3-gpu-a100`. Sketch:

```bash
#SBATCH -A plgttaautopilot3-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1     # one task; torchrun spawns the 8 ranks
#SBATCH --cpus-per-task=128
#SBATCH -t 2-00:00:00
#SBATCH -o logs/%x-%j.out
```

- `plgrid-now` schedules fast and doesn't burn the main allocation, but its QOS caps you at
  **1 GPU / 16 CPUs / 125 GB** — single-GPU smoke tests only. Anything multi-GPU must go to
  `plgrid-gpu-a100`, where a 15-minute test job is cheap.
- Max wall time on `plgrid-gpu-a100` is **2 days**, so training must checkpoint and resume.
  Write resumable state (model, optimizer, sampler epoch, step) and support `--resume`.
  Use `8k3-gpu` (14 days) only for genuinely long runs.
- Set `--cpus-per-task` generously and map `DataLoader(num_workers=...)` to it —
  multimodal video decoding starves the GPUs otherwise. Use `pin_memory=True` and
  `persistent_workers=True`.
- `/tmp` is RAM-backed; writing large temporaries there counts against the job's memory.

### NCCL: bind to InfiniBand explicitly

`eth0`/`eth1` are **down** on compute nodes, so NCCL's default interface selection can pick a
dead or virtual interface and hang until timeout. This is the single most likely cause of a
job that allocates GPUs and then does nothing. Always export:

```bash
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO          # keep while bringing up; drop once stable
```

Do **not** set `NCCL_IB_DISABLE=1` — IB is present and is the fast path. For `torchrun`:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
     --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py
```

If rendezvous hangs on multi-node, resolve `MASTER_ADDR` to the node's `ib0` address instead
of its hostname — verify which network the hostname resolves to before assuming.

## Code conventions

- **Rank gating.** W&B init/logging, dataset downloads, checkpoint writes, and progress bars
  run on rank 0 only. Concurrent writes from 8 ranks to shared Lustre corrupt files and are a
  common source of confusing failures. Gate with a `is_main_process()` helper, not ad-hoc
  `if rank == 0` scattered everywhere.
- **DistributedSampler.** Required for correct sharding, and `sampler.set_epoch(epoch)` must
  be called every epoch or every rank reproduces the same shuffle each epoch.
- **Phase separation** (per `PROJECT.md`): Phase 1 is a raw `torch.distributed` loop
  (`init_process_group(backend="nccl")`, manual DDP) to make the communication layer legible;
  Phase 2 moves to Accelerate + `lerobot` + PEFT. Keep Phase 1 code intact when Phase 2 lands
  — it is part of the deliverable, not scaffolding to delete.
- Barrier before teardown and call `destroy_process_group()`, or jobs hang at exit and waste
  allocation.
- Log effective batch size explicitly (per-device × grad accumulation × world size); it is
  the number that makes W&B curves comparable across runs.

## Where Phase 2 goes

Add `configs/` for accelerate/FSDP/DeepSpeed yaml and hyperparams; keep `slurm/` one
`.sbatch` per experiment shape. `logs/`, checkpoints, venvs, and datasets are gitignored —
none of it belongs in a 10 GB home directory or in the published repo.

## Gotchas

- The A100s are **40 GB**. Any config or guide written for 80 GB A100s/H100s will OOM.
- Pass `device_id=` to `init_process_group`; otherwise `barrier()` infers a device from
  ambient context and can pick the wrong GPU.
- Gradient checkpointing under DDP needs `use_reentrant=False` — the reentrant path misses
  DDP's gradient hooks.
- On multi-node, `torchrun` may log `client socket cannot be initialized ... Address family
  not supported`. Harmless if rendezvous then succeeds; only chase it if the job actually
  hangs, in which case use the node's `ib0` address as `MASTER_ADDR`.
- Keep package caches out of `$HOME` (10 GB): `UV_CACHE_DIR`, and `micromamba -r` for conda
  packages. A stray `conda create` put 3.2 GB in `~/.conda/pkgs` and pushed the quota past
  90%, at which point **SLURM refuses to accept jobs** — `sbatch` fails with "Storage quota
  of $HOME exceeded", not a disk error. Check with `hpc-fs`.
- GPU OOM on a job that previously fit usually means a longer trajectory or larger image
  batch, not a leak — check the data before rewriting the sharding strategy.
