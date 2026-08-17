# CLAUDE.md

Distributed fine-tuning of a Vision-Language-Action (VLA) model for robotic manipulation,
on the Athena GPU cluster (Cyfronet / PLGrid). Full spec: `md_files/PROJECT.md`.

Goal: LoRA fine-tune **π0** (~3B: PaliGemma backbone + flow-matching action expert) on the
**LIBERO** manipulation benchmark, mapping multi-camera frames + language commands to
continuous actions. Deliverables: adapters on HF Hub, a public W&B report, and the SLURM +
distributed training code.

`PROJECT.md` names OpenVLA-7B / SmolVLA; **π0 supersedes that choice** (decided with the user,
2026-08-16). It is the architecture the field converged on and is a first-class `lerobot`
policy, so it reuses the dataset pipeline directly. SmolVLA at 450M is too small for sharding
to be load-bearing; OpenVLA-7B is not a `lerobot` policy and would need a separate RLDS path.

## Current state

**Phase 1 is implemented and verified on the cluster. Phase 2 is in progress** — the
environment is repinned and verified for π0/LIBERO; the training entrypoint is not written yet.

```
train.py              # Phase 1 loop: DDP, grad accum, bf16, checkpoint/resume, W&B
src/dist_utils.py     # process group setup, rank gating, count-weighted reductions
src/toy.py            # VLA-shaped synthetic data + small model (Phase 1 only)
src/runtime.py        # preload_npp() — must run before any torchcodec import
scripts/verify_env.py # checks torch/torchvision/torchcodec/lerobot/peft on a compute node
scripts/probe_pi0.py  # one pi0+LoRA forward/backward: batch keys, VRAM, param counts
slurm/train.sbatch    # 8-GPU production job
slurm/smoke.sbatch    # short 1-GPU + 4-GPU + resume smoke test
pyproject.toml        # dependency declarations + CUDA index pins
uv.lock               # exact pinned versions — commit this
```

`train.py` runs identically at world size 1 (`python train.py`) and under `torchrun`, so
develop on 1 GPU and scale unchanged. Verified on Athena (job 2948190, on the current cu130
stack): 1 GPU, 4 GPUs via torchrun, and resume-from-checkpoint on both. `effective_batch` is
logged on startup.

`train_pi0.py` mirrors that structure with the real model: same `src/dist_utils` scaffolding,
same sampler/no_sync/atomic-checkpoint discipline, but pi0 + LoRA and `LeRobotDataset`
instead of `src/toy.py`.

Run the relevant smoke test after any change to the distributed path:

```bash
sbatch slurm/smoke.sbatch      # Phase 1: 1 GPU + 4 GPU + resume, ~2 min
sbatch slurm/pi0_smoke.sbatch  # Phase 2: same three phases with pi0
```

### Prerequisite: gated PaliGemma repo

The account behind `HF_TOKEN` must accept the licence for
**<https://huggingface.co/google/paligemma-3b-pt-224>** — π0's tokenizer loads from that
gated repo, so processor construction dies with a 403 even though the π0 weights
(`lerobot/pi0`) are themselves ungated. **Already accepted for this project's account**;
a new contributor has to do it again.

### Validation loss is a stochastic estimate — treat small differences as noise

pi0 is a **flow-matching** model: each forward samples a diffusion timestep and noise, so
val loss varies run to run on identical weights. Measured on `adapter_best` over 5 seeds:
**0.3343 +- 0.0015** (spread 0.0042). Consequences:

- **Never report a bare val loss.** Use `scripts/eval_adapter.py --repeats 5`, which seeds
  and reports mean/std/spread.
- **Training-time val loss is noisier than single-process eval** and scattered +-0.007 from
  the true value in run 2988297. Cause: all ranks share `torch.manual_seed(args.seed)` and
  run identical op sequences, so they drew *identical* timestep sequences — correlating the
  noise and inflating variance by ~sqrt(world_size). `evaluate()` now offsets the seed per
  rank and restores RNG state afterwards. Re-measure before trusting fine epoch rankings.
- **Differences below ~0.005 in a training log are not meaningful.** The overfitting trend
  (~0.03 across 10 epochs) is far above the floor and remains solid.

### Results so far (LIBERO-spatial, `--lora-target attention`, r=32)

Run 2988297: 7 epochs, 8 GPUs, 67 min, 89.3 samples/s, 11.7 GiB peak. Reproduced run
2948424's val curve epoch-for-epoch (same seed => same result).

| adapter | reloaded val loss (n=5) |
| --- | --- |
| `adapter_best` (epoch 5) | **0.3343 +- 0.0015** |
| `adapter` (final epoch) | 0.3398 +- 0.0013 |

The ~4 sigma gap means best-checkpoint selection is doing real work — **publish
`adapter_best/`, never `adapter/`**. The saved adapter is bit-exact against `best.pt`
(298/298 tensors, max abs diff 0.0), so the artefact is faithful.

Note **2 of 298 adapter tensors are all-zero** (untrained): the `lora_B` pair on the final
language-model `o_proj`, the same ones DDP reports as receiving no gradient. They are inert,
not broken.

### Measured facts about pi0 + LIBERO

Verified end-to-end in job 2948291 (smoke: 1 GPU → resume → 4 GPU via torchrun, all passed,
including resume *across a world-size change*).

- **Peak VRAM on one A100-40GB at per-device batch 8: 9.8 GiB with gradient checkpointing,
  21.6 GiB without.** Checkpointing cost roughly 13% step time in that measurement.
- **Sharding is not needed.** LoRA freezes the 4.03B base, so in bf16 the weights are ~8 GiB
  and optimizer state is allocated only for the ~2.8M adapter params. FSDP/DeepSpeed ZeRO
  would shard almost nothing here — they become necessary for *full* fine-tuning or a 7B+
  backbone, not for this configuration. Don't add them without a measurement that justifies it.
- **Scaling is near-linear** (job 2948388, effective batch held at 128, `--grad-checkpointing`,
  workers scaled per rank). Peak VRAM stayed 11.3 GiB at every world size, as expected for
  DDP. So multi-GPU here is a genuine wall-clock win, not idle silicon:

  | GPUs | samples/s | elapsed | speedup | efficiency |
  | --- | --- | --- | --- | --- |
  | 1 | 12.9 | 149.4 s | 1.00x | 100% |
  | 2 | 24.8 | 77.5 s | 1.92x | 96% |
  | 4 | 46.8 | 41.0 s | 3.63x | 91% |

- **LoRA on the VLM tower requires `find_unused_parameters=True`** (now the default in
  `train_pi0.py`; `--no-find-unused-parameters` opts out). π0 consumes only the *prefix KV
  cache* from PaliGemma, so the final language-model layer's `o_proj` output feeds nothing
  and its adapter gets no gradient — exactly 2 of 298 trainable tensors. DDP then aborts
  with `Expected to have finished reduction in the prior iteration`. This does not occur
  with `--lora-target default`, which adapts only the action expert.

  **Process lesson: changing `--lora-target` changes which parameters reach the DDP reducer,
  so it invalidates any prior smoke test.** Job 2948411 died on 8 GPUs because the smoke run
  had used different targets. `slurm/pi0_smoke.sbatch` now pins the same
  `--lora-target attention` as production; keep them in sync.

- **The full run is hours, not days.** ~52.5k train samples/epoch; 10 epochs on 8 GPUs is
  roughly 2 hours at the extrapolated rate. The 2-day wall clock in `pi0_train.sbatch` is a
  large over-request — trimming it will schedule sooner. There is ample budget to train on
  all four LIBERO suites (1,693 episodes / 273k frames) instead of `spatial` alone; that
  needs `MultiLeRobotDataset` from `lerobot.datasets.multi_dataset` (note: it is *not*
  exported from `lerobot_dataset`).

- π0 is **4.03B** parameters. `--lora-target` selects which modules get adapters; presets and
  their measured cost at r=32, per-device batch 16, with gradient checkpointing:

  | preset | trainable | % | peak VRAM | samples/s (1 GPU) |
  | --- | --- | --- | --- | --- |
  | `default` (lerobot's own: action-expert q/v only) | 2.77M | 0.069% | 11.3 GiB | 5.0 |
  | `attention` (**production default**) | 12.65M | 0.313% | 11.6 GiB | 4.7 |
  | `attention-mlp` | 53.35M | 1.307% | 13.2 GiB | 3.8 |

  The **vision tower is excluded from every preset on purpose** — SigLIP features transfer
  well across manipulation tasks, and adapting them on 428 episodes risks damaging the part
  doing the most generalisation work. PEFT matches `target_modules` strings with
  `re.fullmatch`, so a typo'd regex matches nothing and silently trains ~0 parameters;
  `train_pi0.py` raises if fewer than 1M params end up trainable.
- `lerobot/libero_spatial_image` is **432 episodes / 52,970 frames / 10 tasks**, and stores
  frames as **PNG images, not video**. So torchcodec is not on this dataset's hot path, but
  dataloading is CPU-bound PNG decode over ~106k small files on Lustre — give it workers.
- Val must be split **by episode**, not by frame: consecutive frames are near-duplicates, so
  a frame-level split leaks nearly the whole val set into training.

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

Installed and verified (Python 3.12.12): torch 2.11.0+cu128, torchvision 0.26.0+cu128,
torchcodec 0.11.1+cu128, transformers 5.5.4, accelerate 1.14.0, peft 0.20.0, lerobot 0.6.1,
scipy 1.18.0, numpy 2.2.6. Verify with `python scripts/verify_env.py` **on a compute node**.

Four pins in `pyproject.toml` are load-bearing and are commented there — don't "simplify" them:

- **lerobot 0.6.x, not 0.4.x.** Two separate reasons, either alone sufficient:
  (a) lerobot fetches hub datasets by *git tag matching its own `CODEBASE_VERSION`*, not from
  `main` — 0.3.x is `v2.1` and the LIBERO ports are `v3.0`-only, so they are silently
  unreachable; (b) 0.4.x's π0 imports `transformers.models.siglip.check`, which exists only on
  the HF branch `fix/lerobot_openpi`. That branch was rebased onto transformers 5.x, violating
  0.4.x's own `transformers<5` cap, so **0.4.x + π0 cannot be resolved at all**. 0.5+ uses
  stock transformers. 0.5+ requires Python 3.12, which is why the venv is 3.12.
- **cu128.** lerobot 0.6.1 caps `torch<2.12` and `torchcodec<0.12`; cu128 carries torch 2.11 /
  torchvision 0.26 / torchcodec 0.11. A100 (sm_80) is fully supported on CUDA 12.8.
  (An earlier cu130 stack existed for lerobot 0.3.2 — that is history, don't restore it.)
- **torch, torchvision and torchcodec all come from the same index.** A PyPI torchvision
  against an index torch fails at import with `operator torchvision::nms does not exist`.
- **`nvidia-npp-cu12`.** torchcodec links NVIDIA NPP (`libnppicc.so.12`) for GPU colour
  conversion; torch does not ship NPP. Without it every `libtorchcodec_core*.so` fails to
  dlopen. `src/runtime.py:preload_npp()` loads it `RTLD_GLOBAL` — **entrypoints must call it
  before importing lerobot or torchcodec**.

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

## Next steps (two candidates, independent — either can be done first)

Both are open as of the `LoRA-based DDP training` commit. They serve different deliverables,
so the choice is about which gap matters more, not which is technically better.

### A. Simulator rollouts, for numbers comparable to the literature

**Gap it closes:** we report a flow-matching *validation loss* (0.3343 +- 0.0015). Published
pi0 / OpenVLA / EO-1 LIBERO results are **task success rates** from closed-loop rollouts.
These are not convertible — a lower val loss does not establish a better policy, and the
model card cannot honestly be compared to published tables without rollouts. This is the
single biggest credibility gap in the current results.

**What it needs:**
- `lerobot[libero]` (adds `hf-libero`; pulls the LIBERO sim + MuJoCo). Currently excluded
  from `pyproject.toml` on purpose — re-lock and re-verify when adding.
- Headless GL on compute nodes: expect to set `MUJOCO_GL=egl` (or `osmesa`) and confirm EGL
  works under SLURM. This is the likely time sink, not the policy code.
- Rollouts use `policy.select_action`, not `forward`; `PI0Config.n_action_steps` controls how
  much of the predicted 50-step chunk is executed per inference.
- Rollouts are sequential per episode and CPU/sim-bound, so parallelise across *episodes* by
  rank rather than trying to batch them.

**Deliverable impact:** makes the HF model card and the W&B report defensible. Do this first
if the portfolio's ML-results story matters more than its systems story.

### B. Full fine-tuning under FSDP / DeepSpeed ZeRO, for the sharding story

**Gap it closes:** `PROJECT.md` asks for FSDP / DeepSpeed ZeRO, but **LoRA makes sharding
unnecessary here** — measured 11.7 GiB peak on a 40 GB card (see above). Shipping sharding
configs for the LoRA run would be untested decoration. Full fine-tuning is the honest way to
make sharding load-bearing:

| component (4.03B, AdamW mixed precision) | memory |
| --- | --- |
| bf16 weights | ~8 GiB |
| bf16 gradients | ~8 GiB |
| fp32 optimizer states (m, v) | ~32 GiB |
| fp32 master weights | ~16 GiB |
| **total before activations** | **~64 GiB** |

That OOMs a 40 GB A100; ZeRO-2 across 8 GPUs lands near 8 + 7 = ~15 GiB plus activations.

**What it needs:** an accelerate-launched variant of `train_pi0.py` (the current one is native
DDP), plus `configs/` for ZeRO-2, ZeRO-3 and FSDP `SHARD_GRAD_OP`. Frame it as a **scoped
systems experiment**: demonstrate OOM on 1 GPU, then fits under each strategy, with a measured
memory-and-throughput table. A few hundred steps is enough — convergence is not the point.

**Do NOT make this the production recipe.** Full FT of 4B params on 392 episodes invites
catastrophic forgetting of pi0's pretraining, and it breaks deliverable #2 (a ~16 GB model
instead of an 11 MB adapter). LoRA stays the production run.

### Also outstanding, smaller

- **Push `adapter_best/` to the Hub with a model card** (deliverable #2). Report val loss
  *with* its std, and state plainly that no rollout success rate has been measured yet.
- **More data beats more tuning on this split.** All four LIBERO suites is 1,693 episodes vs
  392, for ~9 h of compute we demonstrably have; it is the strongest lever against the
  overfitting documented above. Needs `MultiLeRobotDataset` from
  `lerobot.datasets.multi_dataset`.
- Untouched regularisation knobs if staying on `spatial`: `--weight-decay` (currently 0.0),
  `--lr` (1e-4 vs pi0's own 2.5e-5 default), `--lora-r`. Try the first two before rank.

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
