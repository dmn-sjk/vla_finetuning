# CLAUDE.md

Distributed fine-tuning of a Vision-Language-Action (VLA) model for robotic manipulation,
on the Athena GPU cluster (Cyfronet / PLGrid). Full spec: `md_files/PROJECT.md`.

Goal: LoRA fine-tune a VLA model (OpenVLA-7B / SmolVLA) on teleoperated robot trajectory
data (LIBERO, ALOHA, PushT), mapping multi-camera frames + language commands to continuous
actions. Deliverables: adapters on HF Hub, a public W&B report, and the SLURM + distributed
training code.

## Current state

The repo is **greenfield** — the only content is `md_files/PROJECT.md` (a git submodule-ish
clone with its own `.git`). No training code, configs, or SLURM scripts exist yet. Nothing
below describes existing code; it describes the conventions to follow when creating it.

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
| Home | `/net/people/plgrid/plgdmnsjk` — **10 GB quota, ~7.6 GB already used** |
| Node-local disk | None (`TmpDisk=0`); `/tmp` is a 15 GB tmpfs that consumes RAM |
| Internet on compute nodes | **Yes** — HF/W&B reachable from inside jobs |
| Site PyTorch modules | Stale (max 1.13.1) — do not use; build a venv instead |

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

10 GB home with 7.6 GB used means a single HF model download fills it. **Always** redirect
caches to `$SCRATCH` before any download:

```bash
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/.triton
export WANDB_DIR=$SCRATCH/wandb
```

Put venvs, datasets, and checkpoints under `$SCRATCH` too. Only code lives in home.

## Environment

Use `uv` (0.10.2, already installed at `~/.local/bin/uv`). Site PyTorch modules are too old
for modern `lerobot`/`peft`/FSDP; use pip wheels with bundled CUDA and skip the CUDA module
unless something needs `nvcc`.

```bash
export UV_CACHE_DIR=$SCRATCH/.uv-cache
uv venv $SCRATCH/venvs/vla --python 3.11
source $SCRATCH/venvs/vla/bin/activate
uv pip install torch --torch-backend=auto
uv pip install lerobot accelerate peft transformers wandb
```

Never run training on the login node — it has GPUs in `sinfo` but jobs belong on compute nodes.

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

- Use `-p plgrid-now` (1 h limit) for smoke tests — it schedules fast and doesn't burn the
  main allocation on a typo.
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

## Suggested layout

```
train.py                # entrypoint, phase-agnostic
src/                    # data, model wrapping, LoRA config, dist utils
configs/                # accelerate/FSDP/DeepSpeed yaml + training hyperparams
slurm/                  # *.sbatch, one per experiment shape
logs/                   # SLURM stdout (%x-%j.out); gitignored
md_files/PROJECT.md     # spec (separate git repo — don't commit into it)
```

`logs/`, checkpoints, venvs, and datasets must be gitignored — none of it belongs in a 10 GB
home directory or in the published repo.

## Gotchas

- The A100s are **40 GB**. Any config or guide written for 80 GB A100s/H100s will OOM.
- `md_files/` contains its own `.git`; the project root is not a git repo yet. Run
  `git init` at the root before committing, and don't nest the two.
- GPU OOM on a job that previously fit usually means a longer trajectory or larger image
  batch, not a leak — check the data before rewriting the sharding strategy.
