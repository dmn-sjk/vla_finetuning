# Distributed Fine-Tuning of a Vision-Language-Action Model

End-to-end distributed training pipeline for fine-tuning a Vision-Language-Action (VLA) model
for robotic manipulation, built for a multi-GPU SLURM cluster. The model maps multi-camera
visual input and a natural-language instruction to continuous robot control actions.

The project is deliberately split into phases:

- **Phase 1 — distributed systems foundation.** A native `torch.distributed` training loop:
  NCCL process groups, `DistributedSampler` sharding, rank-gated I/O, gradient accumulation
  with sync control, bf16, gradient checkpointing, and resumable checkpointing. *Implemented
  and verified.*
- **Phase 2 — multimodal scaling.** LoRA fine-tuning of **π0** (~4B: a PaliGemma backbone
  plus a flow-matching action expert) on the **LIBERO** manipulation benchmark via
  `lerobot` and `peft`. *Implemented and verified; this is the production recipe.*
- **Phase 2B — sharded full fine-tuning.** All 4.03B parameters trainable, sharded with
  **FSDP** through `accelerate` (ZeRO-2 and ZeRO-3 equivalents). Full FT needs ~64 GiB before
  activations and OOMs a 40 GB card outright, which is what makes sharding load-bearing here
  rather than decorative. *Implemented and measured.*

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
train.py               # Phase 1 entrypoint (world size 1 or N, unchanged)
train_pi0.py           # Phase 2 entrypoint: pi0 + LoRA on LIBERO, native DDP
train_pi0_fsdp.py      # Phase 2B entrypoint: pi0 full FT under FSDP, via accelerate
src/dist_utils.py      # process group setup, rank gating, cross-rank reductions
src/toy.py             # VLA-shaped synthetic data + small model (Phase 1)
src/runtime.py         # native-library preloading (NPP) needed before torchcodec
scripts/verify_env.py  # checks the pieces that have historically broken here
scripts/probe_pi0.py   # one pi0+LoRA forward/backward: batch keys, VRAM, param counts
scripts/eval_adapter.py# seeded, repeated val loss for an adapter or a full checkpoint
slurm/train.sbatch     # Phase 1: 8-GPU production job
slurm/smoke.sbatch     # Phase 1: 1-GPU + 4-GPU + resume smoke test
slurm/pi0_train.sbatch # Phase 2: 8-GPU production job
slurm/pi0_smoke.sbatch # Phase 2: 1-GPU + resume + 4-GPU smoke test
slurm/pi0_fsdp_preflight.sbatch # Phase 2B: 1-GPU checks on the fast queue
slurm/pi0_fsdp_smoke.sbatch     # Phase 2B: 4-GPU FSDP + both resume paths
slurm/pi0_fsdp_bench.sbatch     # Phase 2B: sharding memory/throughput sweep
slurm/pi0_full_train.sbatch     # Phase 2B: full-FT convergence run
slurm/pi0_fsdp_verify_export.sbatch  # reload an exported checkpoint and re-measure it
slurm/pi0_compare_checkpoints.sbatch # LoRA vs full FT under one eval protocol
TODO.md                # what is NOT done, in dependency order
pyproject.toml         # dependencies + CUDA wheel index pins
uv.lock                # exact pinned versions
```

Each phase's entrypoint is kept rather than overwritten. Phase 1's `train.py` is its own
deliverable — the native `torch.distributed` loop is the point of that phase — and Phase 2's
`train_pi0.py` stays the verified production recipe. `train_pi0_fsdp.py` builds beside it and
imports its dataset split, loaders and evaluation function, so a full-FT number and a LoRA
number are measured on exactly the same held-out episodes.

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

### Phase 1 (synthetic model, distributed scaffolding)

Smoke test — single GPU, 4 GPUs via `torchrun`, and checkpoint resume, in about two minutes:

```bash
sbatch slurm/smoke.sbatch
```

Full run on 8 GPUs:

```bash
sbatch slurm/train.sbatch
```

### Phase 2 (π0 + LoRA on LIBERO)

```bash
sbatch slurm/pi0_smoke.sbatch    # 1 GPU -> resume -> 4 GPU, ~10 min
sbatch slurm/pi0_train.sbatch    # 8 GPUs, 7 epochs, ~70 min
sbatch slurm/pi0_scaling.sbatch  # 1/2/4 GPU throughput at fixed effective batch
```

Evaluate a saved adapter. **Always pass `--repeats`**: π0 is a flow-matching model, so
validation loss is a stochastic estimate and a single number is misleading.

```bash
python scripts/eval_adapter.py --adapter $SCRATCH/vla_finetuning/runs/pi0_libero/adapter_best \
  --repeats 5
```

Publish `adapter_best/`, never `adapter/` — the latter is the final epoch, which overfits.

### Phase 2B (π0 full fine-tuning, sharded with FSDP)

```bash
sbatch slurm/pi0_fsdp_preflight.sbatch  # 1 GPU on the fast queue: everything but sharding
sbatch slurm/pi0_fsdp_smoke.sbatch      # 4 GPUs: FSDP, resume, cross-world-size resume
sbatch slurm/pi0_fsdp_bench.sbatch      # the memory/throughput sweep across strategies
sbatch slurm/pi0_full_train.sbatch      # 8 GPUs, full FT to convergence
```

`train_pi0_fsdp.py` takes `--finetune full|lora` and `--sharding ddp|no_shard|shard_grad_op|
full_shard` (the last three are ZeRO-0/2/3 under their torch names). The exported checkpoint
is a complete π0 policy directory, so it evaluates through the same script as the adapter:

```bash
python scripts/eval_adapter.py --policy $SCRATCH/vla_finetuning/runs/pi0_full/model_best \
  --repeats 5
```

**LoRA + DDP remains the production recipe.** Full fine-tuning exists to make the sharding
measurable and to give the LoRA result something to be compared against — see below.

Iterating locally on one GPU, without SLURM:

```bash
$SCRATCH/vla_finetuning/venvs/vla/bin/python train.py \
  --epochs 3 --batch-size 32 --grad-checkpointing
```

Startup logs the effective batch size (per-device × gradient accumulation × world size), which
is the number that makes runs comparable across different cluster shapes.

## Results so far

π0 + LoRA (r=32, attention targets on the VLM backbone and action expert: 12.65M trainable,
0.31% of 4.03B) on LIBERO-spatial, 8× A100-40GB:

| metric | value |
| --- | --- |
| **LIBERO-spatial success rate** | **54.2% ± 2.2** (271/500 closed-loop episodes) |
| base π0, zero-shot, same protocol | 0.0% (0/500) |
| best validation loss | **0.3343 ± 0.0015** (5 seeds) |
| base π0, same split, no fine-tuning | 0.8719 ± 0.0048 (3 seeds) |
| throughput | 89.3 samples/s on 8 GPUs (7.0× over 1 GPU, 87% scaling efficiency) |
| peak VRAM | 11.7 GiB per GPU |
| wall clock | 67 min for 7 epochs |

Validation loss bottoms out around epoch 5 and then rises while training loss keeps falling —
12.65M adapter parameters exceed what 392 episodes support. The best checkpoint is measurably
better than the last (~4σ), which is why best-tracking exists.

### Why sharding is needed at all (Phase 2B)

LoRA fits comfortably on one card, so FSDP would be decoration for that run. Full fine-tuning
is a different problem. All 4.028B parameters trainable, one A100-40GB, per-device batch 8,
gradient checkpointing already on:

```
OOM_EXPECTED finetune=full sharding=ddp world_size=1 per_device_bs=8
             grad_checkpointing=True peak_vram_gib=37.9 capacity_gib=39.7
```

fp32 weights (16.1 GiB) and fp32 gradients (16.1 GiB) fill the card before AdamW allocates
its 32 GiB of moments. FSDP shards the optimizer state and gradients (`shard_grad_op`, the
ZeRO-2 equivalent) or the parameters as well (`full_shard`, ZeRO-3).

Measured across strategies (per-device batch 8, gradient checkpointing unless noted, 30
timed steps):

| GPUs | strategy | result | samples/s | peak VRAM |
| --- | --- | --- | --- | --- |
| 1 | unsharded | **OOM** | — | 37.4 GiB of 39.7 before dying |
| 2 | FULL_SHARD | **OOM** | — | — |
| 8 | unsharded | **OOM** | — | — |
| 4 | SHARD_GRAD_OP (ZeRO-2) | runs | 18.6 | 25.1 GiB |
| 4 | FULL_SHARD (ZeRO-3) | runs | 19.3 | 25.1 GiB |
| 8 | SHARD_GRAD_OP (ZeRO-2) | runs | 48.4 | 17.0 GiB |
| 8 | FULL_SHARD (ZeRO-3) | runs | **49.7** | 17.0 GiB |
| 8 | FULL_SHARD, *no* grad ckpt | runs | 31.0 | 33.2 GiB |
| 8 | FULL_SHARD, **LoRA** | runs | 50.9 | 11.7 GiB |

Sharding here is not a speedup — it is the difference between a configuration that does not
run and one that does, and **four GPUs is the floor**. Note that **eight GPUs on DDP still
OOMs**: DDP replicates the optimizer state on every rank, so adding hardware does not help,
only sharding does. Both strategies produced identical validation loss (0.4532) at the same
step and seed, which is the check that sharding changed the memory layout and not the math.

Three things in that table are worth a second look:

- **ZeRO-3 reaches the same peak as ZeRO-2, not a lower one.** 39% of the model sits in the
  root FSDP unit — a direct consequence of the wrap constraint below — and stays gathered
  across forward and backward, so ZeRO-3 cannot shard the part that dominates. (This was
  checked against a measurement artefact: the fp32 model is built on the GPU before sharding,
  costing 15.5 GiB, but that is below the reported peaks, so they are genuine training peaks.)
- **Gradient checkpointing is 60% *faster* here, not slower** (49.7 vs 31.0 samples/s) — the
  opposite sign to the LoRA run's ~13% cost. At 33.2 GiB on a 40 GiB card the allocator is
  under enough pressure that recomputation wins back more than it spends.
- **LoRA under FSDP peaks at 11.7 GiB, exactly matching the LoRA+DDP production run** — two
  independent code paths agreeing on the same number.

### What full fine-tuning actually bought

The point of running it to convergence was to replace an assumption with a number. Both
artefacts were then re-evaluated with one script, one batch size and one set of seeds on the
same 40 held-out episodes — the two training logs are not directly comparable to each other:

| checkpoint | success rate | val loss (n=3) | best at | trainable | artefact | min GPUs |
| --- | --- | --- | --- | --- | --- | --- |
| base π0 | 0.0% | 0.8719 ± 0.0048 | — | — | — | — |
| LoRA `adapter_best` | **54.2% ± 2.2** | **0.3344 ± 0.0020** | epoch 5 of 7 | 12.65M | 49 MB | 1 |
| full FT `model_best` | **56.6% ± 2.2** | **0.3350 ± 0.0014** | epoch 0 of 5 | 4,028M | 7.6 GB | 4 |

**Two independent metrics agree.** The success-rate gap is +2.4 points with SE 3.1
(z = 0.76); the val-loss gap is 0.0006, inside the ±0.002 seed noise. Both say
**statistically indistinguishable**. LoRA reaches the same quality with 0.3% of the trainable
parameters, a ~160× smaller artefact, and a quarter of the hardware.

They are not the same policy, though. They tie in aggregate while differing per task, with
two differences surviving Bonferroni across the 10 tasks: task 5 "on the ramekin" (LoRA 30%
vs full FT 6%, z = −3.29) and task 9 "on the wooden cabinet" (34% vs 62%, z = +2.92). Their
strengths cancel — something no loss comparison could show.

Success rates come from 500 closed-loop episodes per checkpoint on the standard LIBERO
protocol (10 tasks × 50 trials from the suite's saved initial states, 280-step cap). **Do not
read 54.2% against the ~96% published for π0 on LIBERO-spatial**: that is full fine-tuning
with openpi's recipe, not a 67-minute 12.65M-parameter LoRA. Equally, the 0.0% base row is a
genuine zero-shot floor — the released `lerobot/pi0` never saw LIBERO — not the paper's
number.

Full fine-tuning reaches LoRA's quality in a single epoch and then collapses:

| epoch | 0 | 1 | 2 | 3 | 4 |
| --- | --- | --- | --- | --- | --- |
| val loss | **0.3344** | 0.3658 | 0.5015 | 0.6869 | 0.8361 |
| train loss | ~0.35 | ~0.31 | 0.24 | 0.15 | 0.084 |

By epoch 4 the model is back at *base-model* validation quality (0.8361 vs 0.8719 untrained)
with train loss at 0.084 — catastrophic forgetting, measured rather than asserted. Updating
all 4.028B parameters buys no generalization over updating 12.65M of them, costs a 7.6 GB
checkpoint instead of a 49 MB adapter, and needs a minimum of four GPUs where LoRA fits on
one. **LoRA + DDP remains the production recipe** — now for a measured reason rather than an
assumed one.

One finding worth stating plainly: **the textbook FSDP wrap policy is wrong for π0.** FSDP
unshards parameters in a module's pre-forward hook, but π0 never calls its decoder layers'
`forward` — it hand-rolls a fused loop so the VLM prefix and the action expert can share one
attention op, reaching into `layer.self_attn.q_proj(...)` directly. Wrapping
`GemmaDecoderLayer` would build units whose hooks never fire, leaving parameters sharded
while the math runs: a silent correctness bug rather than a clean failure. The policy wraps
`GemmaMLP` and `SiglipEncoderLayer` instead, which *are* invoked as modules, covering 2.45B
of 4.04B parameters; the rest lands in the (still sharded) root unit. The same hook mechanism
is why DeepSpeed ZeRO-3 cannot be used on this model at all.

## Next steps

`TODO.md` is the authoritative list of what is not done. Nothing is blocked: the `$HOME`
quota is clear and Phase 2B is committed. `CLAUDE.md` has the detailed engineering notes.

**A — Simulator rollouts.** *Done*, see the success rates above. `scripts/eval_libero.py`
runs the standard LIBERO protocol; `scripts/diagnose_libero.py` exists because the first
attempt scored 0% on a known-good checkpoint (the post-processor was never given the
dataset's normalization stats, so rollouts drove the arm with z-scores — a validation loss
cannot catch that, because computing one never touches the inference path).

**B — Full fine-tuning under FSDP.** *Done*, see `train_pi0_fsdp.py` above.

Remaining: publish `adapter_best/` and `model_best/` to the Hub — cards are written
(`model_cards/`) and `scripts/publish_to_hub.py` applies them; evaluate both checkpoints on a
*different* LIBERO suite as an out-of-distribution test (~1 GPU-hour, no retraining, and the
per-task divergence above makes it a real question); train on all four LIBERO suites (1,693
episodes vs 392) as the strongest lever against the overfitting; and the untouched
regularisation knobs (`--weight-decay` is 0.0, `--lr` is 1e-4 against π0's own 2.5e-5
default).
