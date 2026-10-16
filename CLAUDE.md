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

**Phase 2 (LoRA) and Phase 2B (full FT under FSDP) are implemented, verified on the
cluster, fully measured and published** (`dmnsjk/pi0-libero-spatial-lora`,
`dmnsjk/pi0-libero-spatial-full`). Outstanding work is in `TODO.md` and `IMPROVEMENTS.md`;
nothing below is in flight.

The repository was cleaned down to the code that produced and evaluated the published
checkpoints. Phase 1 (`train.py`, `src/toy.py`), the smoke/preflight jobs, `probe_pi0.py`,
`diagnose_libero.py` and the recheck/sanity/diagnose/verify-export jobs were removed; all of
them are in git history. Findings below that cite those tools are still valid. (Phase 1 was
verified on Athena in job 2948190: 1 GPU, 4 GPUs via torchrun, and resume on both.)

```
train_pi0.py           # Phase 2 production: pi0 + LoRA, native DDP  <- the production recipe
train_pi0_fsdp.py      # Phase 2B: pi0 full FT, accelerate + FSDP; also --finetune lora
src/dist_utils.py      # process group setup, rank gating, count-weighted reductions
src/runtime.py         # preload_npp() — must run before any torchcodec import
src/paths.py           # $VLA_ROOT / $VLA_RUNS resolution, shared with the shell
scripts/verify_env.py  # checks torch/torchvision/torchcodec/lerobot/peft on a compute node
scripts/eval_adapter.py# seeded, repeated val loss for an adapter *or* a full checkpoint
scripts/eval_libero.py # closed-loop LIBERO rollouts (success rate); --verify-obs
scripts/render_results.py # card tables from results/*.json; --check re-derives every rate
scripts/publish.sh     # publish both checkpoints to the Hub (dry run unless --go)
scripts/publish_to_hub.py # upload one checkpoint with its card from model_cards/
slurm/common.sh        # shared job prologue: paths, caches, .env, venv activation
slurm/pi0_train.sbatch # Phase 2: 8-GPU LoRA production job
slurm/pi0_scaling.sbatch        # Phase 2: 1/2/4-GPU throughput at fixed effective batch
slurm/pi0_fsdp_bench.sbatch     # Phase 2B: the sharding memory/throughput sweep
slurm/pi0_full_train.sbatch     # Phase 2B: full-FT convergence run
slurm/pi0_compare_checkpoints.sbatch # base vs LoRA vs full FT val loss, one protocol
slurm/pi0_libero_eval.sbatch    # base vs LoRA vs full FT success rate, LIBERO-spatial
slurm/pi0_libero_reference.sbatch # IMPROVEMENTS.md item 1: lerobot's LIBERO pi0, our harness
results/               # per-episode rollout records behind the published success rates
model_cards/           # Hub READMEs of both checkpoints, byte-identical to the Hub copies
checkpoints.sha256     # hashes of the archived checkpoints that were evaluated
TODO.md                # what is NOT done: evaluation gaps, housekeeping
IMPROVEMENTS.md        # the gap to published pi0 LIBERO numbers: causes, costs, order
pyproject.toml         # dependency declarations + CUDA index pins
uv.lock                # exact pinned versions — commit this
constraints.txt        # pins for installing the LIBERO sim stack into the venv
```

`train_pi0.py` runs identically at world size 1 (`python train_pi0.py`) and under
`torchrun`, so develop on 1 GPU and scale unchanged: `src/dist_utils` scaffolding,
`DistributedSampler`, `no_sync` on accumulation micro-batches, atomic checkpoints, and
`effective_batch` logged on startup.

There are no smoke-test job scripts any more. After any change to the distributed path,
re-check before a production run, as the removed jobs did. For `train_pi0.py`: 1 GPU, then
resume from that checkpoint, then 4 GPUs via torchrun (resume *across a world-size change*).
For `train_pi0_fsdp.py`, on 1 GPU: LoRA on the DDP path with save + export, resume, and the
guard that refuses a single-process `--sharding full_shard`. Then on 4 GPUs: full FT under
FULL_SHARD (the custom auto-wrap policy fires), resume at the same world size,
SHARD_GRAD_OP + full-model export (the publishable artefact is written), LoRA under FSDP
(mixed `requires_grad` inside a unit), and resume onto 2 GPUs (DCP reshards 4-rank shards).
That last one uses LoRA, because full FT does not fit on two cards (job 2989671, below).
FSDP itself has no single-GPU check: accelerate only engages it from a distributed launch.
Do the single-GPU parts on `plgrid-now` first: it schedules in seconds, does not bill the
main allocation, and catches everything that does not need more than one GPU.

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
- **Sharding is not needed *for LoRA*.** LoRA freezes the 4.03B base, so in bf16 the weights
  are ~8 GiB and optimizer state is allocated only for the adapter params. FSDP/DeepSpeed ZeRO
  would shard almost nothing here — they become necessary for *full* fine-tuning or a 7B+
  backbone, not for this configuration. That is what Phase 2B measures; see
  "Full fine-tuning under FSDP" below. Keep LoRA + DDP as the production recipe.
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
  had used different targets. Any smoke run must pin the same `--lora-target attention`
  as production.

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

## Full fine-tuning under FSDP (`train_pi0_fsdp.py`)

Phase 2B. This exists to make sharding *load-bearing* rather than decorative, and to answer
the obvious objection to the LoRA run: LoRA fits in 11.7 GiB, so what is FSDP for?

**It is for the configuration that does not otherwise run at all.** Measured on one
A100-40GB, all 4.028B parameters trainable, per-device batch 8, gradient checkpointing on
(job 2989458):

```
OOM_EXPECTED finetune=full sharding=ddp world_size=1 per_device_bs=8
             grad_checkpointing=True peak_vram_gib=37.9 capacity_gib=39.7
```

fp32 weights (16.1 GiB) + fp32 gradients (16.1 GiB) + activations fills the card before
AdamW ever allocates its 32 GiB of moments. Sharding those three across ranks is the whole
point; `--sharding shard_grad_op` shards gradients and optimizer state (ZeRO-2),
`--sharding full_shard` shards parameters as well (ZeRO-3).

### Measured: what sharding actually buys (jobs 2989458, 2990654)

pi0 4.028B trainable, LIBERO-spatial, per-device batch 8, gradient checkpointing on unless
noted, `--reduce-dtype fp32`, 30 optimizer steps timed after a 5-step warmup. Throughput
covers training steps only; peak VRAM is the training peak (jobs 2989458, 2990654, 2990653).

| GPUs | strategy | result | samples/s | peak VRAM |
| --- | --- | --- | --- | --- |
| 1 | unsharded (DDP path) | **OOM** | — | 37.4 GiB of 39.7 before dying |
| 2 | FULL_SHARD (ZeRO-3) | **OOM** | — | — |
| 8 | unsharded (DDP path) | **OOM** | — | — |
| 4 | SHARD_GRAD_OP (ZeRO-2) | runs | 17.5 / 18.6 | 25.1 GiB |
| 4 | FULL_SHARD (ZeRO-3) | runs | 19.2 / 19.3 | 25.1 GiB |
| 8 | SHARD_GRAD_OP (ZeRO-2) | runs | 48.4 | 17.0 GiB |
| 8 | FULL_SHARD (ZeRO-3) | runs | **49.7** | 17.0 GiB |
| 8 | FULL_SHARD, *no* grad ckpt | runs | 31.0 | 33.2 GiB |
| 8 | FULL_SHARD, **LoRA** | runs | 50.9 | 11.7 GiB |

(Two numbers in the 4-GPU rows are the same configuration measured in two separate jobs —
they agree to ~5%, which is the reproducibility of these measurements.)

Read this as the honest answer to "what is FSDP for here": not a speedup, but the
difference between a configuration that does not run at all and one that does. For LoRA,
sharding buys nothing (11.7 GiB unsharded on one card) — that is why LoRA stays on DDP.

**Eight GPUs on DDP still OOMs.** DDP replicates the optimizer state on every rank, so
adding hardware does not help at all — only sharding does. That row is the cleanest
statement of what FSDP is buying.

Results worth not glossing over:

- **ZeRO-2's arithmetic predicts the measured peak exactly at both world sizes.** At world
  size N: 16.1/N (fp32 master) + 8.1 (bf16 params, *not* sharded by ZeRO-2) + 16.1/N (grads)
  + 32.2/N (AdamW) + ~1 (activations). That is 25.2 at N=4 (measured 25.1) and 17.1 at N=8
  (measured 17.0). The model is right.
- **ZeRO-3 measured identical to ZeRO-2 at both world sizes** (25.1 and 17.0), and this is
  a real result rather than a measurement artefact. It was worth checking: pi0 is built in
  fp32 and moved to the GPU *before* FSDP shards it, and `max_memory_allocated()` is a
  process-lifetime high-water mark, so a large enough construction transient would floor
  every row and hide differences beneath it. `train_pi0_fsdp.py` now resets the peak after
  `prepare()` and reports `setup_peak_gib` separately — **measured at 15.5 GiB**. Since 15.5
  is below the 17.0 reported at world size 8, the reported peaks are genuine training peaks
  and the ZeRO-2/ZeRO-3 equivalence stands.

  The remaining explanation is the wrap policy: 39% of the model sits in the root FSDP unit
  — a consequence of pi0's fused layer loop preventing per-layer wrapping — and FSDP1 keeps
  the root gathered across forward and backward, so ZeRO-3 cannot shard the part that
  dominates. Still an inference, but now the only one standing.
- **ZeRO-3 is marginally faster than ZeRO-2** (19.3 vs 18.6 at N=4; 49.7 vs 48.4 at N=8)
  despite doing more communication. Small but consistent across both world sizes. There is
  no reason to prefer ZeRO-2 here.
- **Scaling 4 -> 8 GPUs is superlinear** (19.3 -> 49.7 samples/s, 2.6x for 2x the hardware).
  Not magic: at 4 GPUs the 25.1 GiB working set on a 40 GiB card leaves the allocator
  fighting FSDP's all-gather buffers, and halving per-rank state relieves it. Sharding buys
  throughput here by buying headroom.
- **Gradient checkpointing is a throughput *win* here, not a cost.** 8 GPUs, FULL_SHARD,
  per-device batch 8: 49.7 samples/s at 17.0 GiB with it, **31.0 samples/s at 33.2 GiB
  without**. That is the opposite sign to the LoRA run, where it cost ~13% step time. Same
  cause as above — at 33.2 GiB the allocator is under enough pressure that recomputation
  buys back more than it spends. Never carry a checkpointing cost estimate across memory
  regimes. It also puts a number on activations: ~16 GiB at per-device batch 8.
- **LoRA under FSDP peaks at 11.7 GiB, matching the LoRA+DDP production run exactly.** Two
  entirely different code paths agreeing on the same number is the best evidence available
  that both are measuring what they claim. Note its *setup* peak (15.1 GiB) exceeds its
  training peak: for LoRA the card requirement is set by model construction, not training.

**Four GPUs is the floor, not one.** Sharding across two ranks still does not fit: job
2989671 resharded a checkpoint onto 2 ranks correctly and then OOM'd in the first step at
35.4 GiB allocated, 2.94 GiB short of the card, at per-device batch *2*. At world size 4 the
same configuration peaks at 25.1 GiB and trains. This is why the (removed)
`pi0_fsdp_smoke.sbatch` tested cross-world-size resume with LoRA rather than full FT — the
resharding path is what that phase checked, and it must not double as an assertion that full
FT fits on two cards.

### pi0 breaks the textbook auto-wrap policy — this is the interesting part

FSDP unshards parameters in a module's **pre-forward hook**, so a wrapped module only works
if its `forward` is actually called. `PaliGemmaWithExpertModel` never calls its decoder
layers: `compute_layer_complete` (lerobot/policies/pi0/modeling_pi0.py:78) reaches into
`layer.self_attn.q_proj(...)` directly so the VLM prefix and the action-expert suffix can
share one attention op. `GemmaDecoderLayer.forward` and `GemmaAttention.forward` are dead
code in this model.

Consequences, all of them load-bearing:

- **`TRANSFORMER_BASED_WRAP` on `GemmaDecoderLayer` is wrong here.** It would build units
  whose hooks never fire, leaving parameters sharded while the math runs. That is a silent
  correctness bug, not a clean failure.
- We wrap **`GemmaMLP` and `SiglipEncoderLayer`** instead — both are invoked as modules.
  Measured coverage: **2.45B of 4.04B parameters (61%)**, printed as `fsdp_wrapped=` on
  startup. The remaining 39% (embeddings, attention projections, norms) lands in the root
  unit, which is still sharded — it just means one coarser all-gather.
- `train_pi0_fsdp.py` **raises if coverage drops below 50%**, because a transformers upgrade
  that renames those classes would otherwise silently collapse FULL_SHARD into SHARD_GRAD_OP
  and quietly invalidate the memory table.
- **DeepSpeed ZeRO-3 is unavailable for the same reason** (it gathers parameters through
  module hooks too). ZeRO-2 would work, but FSDP's SHARD_GRAD_OP is the same algorithm
  without the extra dependency — which is why no DeepSpeed dependency was added.

### The head-to-head, measured the same way (job 2991314)

Both artefacts re-evaluated with one script, one batch size, one set of seeds, on the same
40 held-out episodes. This is the comparison to quote; the two training logs are *not*
comparable to each other directly (different eval batch sizes, and the LoRA figure predates
the `--repeats` fix).

| checkpoint | val loss (n=3) | trainable | artefact | min GPUs |
| --- | --- | --- | --- | --- |
| base `lerobot/pi0` | 0.8719 +- 0.0048 | — | — | — |
| LoRA `adapter_best` | **0.3344 +- 0.0020** | 12.65M | 49 MB | 1 |
| full FT `model_best` | **0.3350 +- 0.0014** | 4,028M | 7.6 GB | 4 |

The two fine-tunes differ by 0.0006, well inside the +-0.002 seed noise: **statistically
indistinguishable quality.** LoRA gets there with 0.3% of the trainable parameters, a ~160x
smaller artefact, and a quarter of the hardware. That is the whole argument for the
production recipe, and it is now a measurement rather than a prior.

### Full FT reaches LoRA's quality in one epoch, then overfits hard

Job 2991036: 4 GPUs, FULL_SHARD, effective batch 256, lr 2.5e-5 cosine, the *same* 392/40
episode split as the LoRA production run so the curves are comparable.

| epoch | val loss | train loss |
| --- | --- | --- |
| 0 | **0.3344** (best) | ~0.35 |
| 1 | 0.3658 | ~0.31 |
| 2 | 0.5015 | 0.24 |
| 3 | 0.6869 | 0.15 |
| 4 | 0.8361 | 0.084 |

Run completed in 3 h 20 m; `THROUGHPUT ... samples_per_s=21.0 ... peak_vram_gib=25.1` over
11,386 s of timed training, which matches the 30-step benchmark row and confirms the
measurement holds at production length.

Three things follow, and they are the reason this is not the production recipe:

- **Full FT peaks at epoch 0 at parity with LoRA** (0.3344 vs 0.3343 +- 0.0015). Training-time
  val loss carries ~+-0.007 noise, so read this as "both land at ~0.33", not as an exact tie.
  Updating 4.028B parameters buys no generalisation over updating 12.65M of them.
- **It then degrades monotonically** while train loss keeps falling — 392 episodes cannot
  support 4B trainable parameters. LoRA's best was epoch 5 of 7; full FT's is epoch 0 of 5.
- **By epoch 4 it is back at base-model quality** (0.8361 vs the untrained 0.8719 +- 0.0048)
  with train loss at 0.084. That is catastrophic forgetting measured rather than asserted:
  the run memorised the training episodes and gave back everything the pretraining provided.
  Anyone tempted to "just train longer" on this dataset size has the counter-example here.

Cost of the two artefacts for the same result: 7.6 GB vs 49 MB, and a 4-GPU floor vs running
on one. **LoRA + DDP stays the production recipe**, now for a measured reason rather than an
assumed one.

### Reference point: what fine-tuning is worth at all

Base `lerobot/pi0`, no fine-tuning, on the **production** 40-episode val split (job 2991110,
3 seeds, with the `--repeats` fix below). The rows below come from different jobs;
`slurm/pi0_compare_checkpoints.sbatch` has since been extended to measure all three in one
job under one protocol (3 seeds, batch size 8), but that version has not been run yet:

| checkpoint | val loss (40 val episodes) |
| --- | --- |
| base `lerobot/pi0` | **0.8719 +- 0.0048** |
| LoRA `adapter_best` | 0.3343 +- 0.0015 |
| full FT `model_best` | 0.3350 +- 0.0014 (job 2991314; LoRA 0.3344 +- 0.0020 there) |

Quote this whenever a fine-tuned number is quoted. Without it, "val loss 0.33" is
uninterpretable — it is the drop from 0.87 that shows the training did something, and it is
the number that makes the LoRA-vs-full-FT comparison meaningful rather than a bare pair.

Note the base model's spread (0.0048) is ~3x the LoRA adapter's documented 0.0015 on the
same split. Both are real measurements taken with different eval batch sizes, so do not
treat 0.0015 as *the* noise floor for every checkpoint — re-measure per artefact.

### The exported checkpoint is real, and verified end-to-end

`train_pi0_fsdp.py` gathers the FSDP shards and writes a complete pi0 policy directory —
`config.json`, `model.safetensors` (8.06 GB in bf16 for 4.03B params), and both processor
configs including the normalisation statistics. Without those statistics the weights load
and predict nonsense, which is the failure mode worth guarding against.

Verified in job 2991045 by loading the exported directory fresh and re-evaluating:

| | val loss (4 val episodes) |
| --- | --- |
| training log (phase 3 of the removed `pi0_fsdp_smoke.sbatch`) | 0.6796 |
| reloaded export, seeds 0-1 | **0.6850 +- 0.0041** |

Consistent to ~1.3 sigma, so the artefact is what it claims to be. After any change to the
export path, re-check the same way: load the exported directory fresh with
`scripts/eval_adapter.py --policy <dir> --val-episodes <same as the run> --repeats N` and
compare against the training log. The split must match the run that produced the
checkpoint, or "validation" means a different set of episodes and the number is not
comparable to anything (`eval_adapter.py` defaults to the production 40).

**Caveat on that number:** a 4-episode val set (~490 frames) is far noisier than the
40-episode set the LoRA results use. Seed 1234 gives 0.7116 on the same weights — 0.03
away from the seed-0/1 mean. Judge export fidelity on the production val split with
`--repeats 5`, not on a smoke run's.

### `scripts/eval_adapter.py --repeats` was reporting a structural zero

Found while verifying the export above. The script seeded with `torch.manual_seed(seed + i)`
before calling `T.evaluate()`, but `evaluate()` **re-seeds internally** (`seed: int = 1234`,
offset per rank) and so overwrote it. Every repeat therefore evaluated identically and
`std`/`spread` were always exactly 0.0000 — job 2991038 reported `std=0.0000 spread=0.0000`
over two seeds, which is what surfaced it.

Fixed by passing the seed through: `T.evaluate(..., seed=args.seed + i)`. The internal
seeding was added to `evaluate()` *after* the `0.3343 +- 0.0015` measurement was taken, so
that historical number is still valid — but anything measured with `--repeats` between that
change and this fix has a meaningless spread. Re-measure before quoting one.

### Other things that are not obvious

- **The model is built in fp32, not bf16.** FSDP1 flattens each unit into a single tensor
  and refuses to mix dtypes inside it, and pi0's own bf16 mode is deliberately mixed (it
  keeps the vision tower and layernorms in fp32). So `dtype="float32"` plus an FSDP
  `MixedPrecision` policy — bf16 compute and all-gather, fp32 master weights and gradient
  reduction. This is why a LoRA row in this script peaks at 22.7 GiB where `train_pi0.py`
  peaks at 11.6 GiB: it is the same model in fp32.
- **accelerate silently ignores an FSDP plugin at world size 1** — the FSDP branch in
  `accelerate/state.py` sits inside the MULTI_GPU arm. A single-process `--sharding
  full_shard` would train unsharded and report a number that measures nothing, so the
  script raises instead. Verified in the removed `pi0_fsdp_preflight.sbatch` (job 2989458).
- **No `no_sync()` on accumulation micro-batches**, unlike `train_pi0.py`. FSDP's `no_sync`
  keeps *unsharded* gradients resident for the whole accumulation window, handing back
  exactly the memory sharding just saved.
- **The scheduler is not passed through `accelerate.prepare`.** An `AcceleratedScheduler`
  steps once per process per optimizer step, which would multiply the LR schedule by
  world_size. It is stepped by hand and registered for checkpointing instead.
- **`save_state` serialises the whole 4.03B model even in LoRA mode** (16 GB per epoch) —
  it is a resume point, not an artefact. `--no-checkpoint` skips it for benchmark rows.
- Throughput is timed over **training segments only**, summed per epoch. A single
  start-to-finish stopwatch folds each epoch's validation pass and 16 GB checkpoint write
  into the window. Note `train_pi0.py`'s published samples/s does *not* do this, so its
  numbers and this script's are not directly comparable.

## Closed-loop LIBERO evaluation (`scripts/eval_libero.py`)

Validation loss is not the metric the literature reports. pi0 / OpenVLA / EO-1 LIBERO tables
are **task success rates** from closed-loop rollouts, and no validation loss converts to one.
This is what closes that gap.

**Protocol** (the LIBERO/OpenVLA convention, so the numbers are comparable): all 10 tasks of
a suite, 50 trials per task, trial *i* starting from the suite's *i*-th saved initial state
(deterministic — no random resets), 10 no-op settle steps after reset, episode capped at the
suite's own budget (280 steps for spatial) and scored a failure if the goal predicate never
fires. 500 episodes per checkpoint.

### The camera orientation is the trap, and it is silent

`lerobot/libero_spatial_image` is an RLDS port, and the port **rotates the raw robosuite
frame 180 degrees** (robosuite renders origin-bottom-left *and* mirrored). Feed the policy
un-rotated frames and everything runs — no error, no warning, plausible-looking actions — and
it scores near zero for a reason nothing in the logs explains.

`--verify-obs` settles it by measurement rather than by reading source: it resets the env,
builds the observation four ways, and scores each against a real dataset frame. Job 3108456:

| transform | agentview MSE | wrist MSE |
| --- | --- | --- |
| none | 0.09549 | 0.13613 |
| flipud | 0.07114 | 0.08945 |
| fliplr | 0.06378 | 0.12276 |
| **rot180** | **0.00599** | **0.01526** |

An order of magnitude, on both cameras. Run `--verify-obs` again if the dataset ever changes.
The same job's 4-episode smoke rollout ran base `lerobot/pi0` with `flipud` (set up before
this measurement settled the orientation) and scored 0/4, which tells us nothing: base pi0
scores 0% even with rot180 (0/500, job 3108565).

### The observation format is the dataset's, not lerobot's LiberoEnv default

lerobot ships `lerobot/envs/libero.py` and a `LiberoEnv` config, but its feature map does
**not** match the dataset we trained on:

| | lerobot `LiberoEnv` | `lerobot/libero_spatial_image` (what we trained on) |
| --- | --- | --- |
| cameras | `observation.images.image` / `.image2` | `observation.images.image` / `.wrist_image` |
| state | dict of `eef_pos`, `eef_quat`, `eef_mat`, `gripper_qpos`, ... | one 8-D vector `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]` |

So `eval_libero.py` drives `libero.libero.envs.OffScreenRenderEnv` directly and builds the
dataset's format itself. Using lerobot's env config instead would run and score near zero,
the same failure mode as the orientation.

The script also checks each suite task's language string against the training set's task
list and warns on a mismatch — a prompt the policy never saw is a third silent zero.

### Normalization stats come from the *training* dataset, always

`--dataset` names the dataset supplying `meta.stats`, and it defaults to the training set
even when `--suite` points elsewhere. At deployment a policy has no access to the target
suite's statistics; quietly substituting them would flatter any out-of-distribution number.

### LoRA is not merged by default

`merge_and_unload()` would save the adapter's extra matmuls on ~10^5 forward passes, but the
merge happens in the load dtype and in bf16 it rounds — the rolled-out weights would no
longer be the artefact being reported on. Every published number uses the unmerged adapter
(`PeftModel.from_pretrained`); the unused `--merge-lora` opt-in was removed.

### Success rates, measured (job 3108565)

500 episodes per checkpoint on the standard protocol (10 tasks x 50 trials from the suite's
saved initial states, 280-step cap, `n_action_steps=5`). This is the number the literature
reports and the gap TODO.md called the project's biggest credibility problem.

| checkpoint | success rate | successes | val loss (n=3) |
| --- | --- | --- | --- |
| base `lerobot/pi0` zero-shot | 0.0% | 0/500 | 0.8719 +- 0.0048 |
| LoRA `adapter_best` | **54.2% +- 2.2** | 271/500 | 0.3344 +- 0.0020 |
| full FT `model_best` | **56.6% +- 2.2** | 283/500 | 0.3350 +- 0.0014 |

**The success rate independently confirms the val-loss tie.** full FT - LoRA = +2.4 points,
SE 3.1, **z = 0.76** — not significant, matching the 0.0006 val-loss gap that sits inside seed
noise. Two unrelated metrics now say a 0.31%-parameter LoRA equals full fine-tuning of all
4.03B. That was previously an inference from a loss; it is now a measurement of behaviour.

**But they are not the same policy.** They tie in aggregate while differing per task, and two
differences survive Bonferroni across the 10 tasks (|z| > 2.81):

| task | LoRA | full FT | z |
| --- | --- | --- | --- |
| 5, "on the ramekin" | 30% | 6% | -3.29 |
| 9, "on the wooden cabinet" | 34% | 62% | +2.92 |

Their strengths cancel. No loss comparison could have surfaced this.

**Base pi0 scores a clean 0/500.** That is the honest zero-shot floor — the released
`lerobot/pi0` was never trained on LIBERO and has never seen this embodiment or action space.
It is **not** the ~96% associated with pi0 on LIBERO-spatial, which comes from full
fine-tuning with openpi's recipe; do not quote the paper's figure as "the baseline" here, and
do not compare our 54.2% to it either. Different recipe, 0.31% of the parameters, 67 minutes.

`n_action_steps=5` follows openpi's LIBERO replan interval and was **not** tuned here. It is
the most obvious untested lever on these numbers.

Cost: 27 min per 500 episodes on 4 GPUs when everything fails (base, no early termination),
~23 min when roughly half succeed.

### A good validation loss does not mean the policy works — the inference path is untested by it

The first LIBERO evaluation scored **0/15 on `adapter_best`**, a checkpoint with a val loss
of 0.3344 against base pi0's 0.8719. Exactly one of those numbers was wrong about the policy,
and it was the rollout.

**Cause:** `make_pre_post_processors` was given `preprocessor_overrides` but not
`postprocessor_overrides`, so the unnormalizer kept whatever stats `lerobot/pi0` shipped
instead of LIBERO's. Predicted actions reached the simulator as **z-scores** — roughly 25x
too large on the rotation dims — so the arm flailed and nothing ever succeeded. Nothing
raised.

**The tell**, from the removed `scripts/diagnose_libero.py` (job 3108532): predicted gripper
`-1.08` where the expert commands `-1.0`. The dataset's action gripper stats are mean 0.084,
std 0.996, and `(-1 - 0.084) / 0.996 = -1.088`. The output was in normalized units to three
decimals.

**Why no earlier check could have caught it.** Training only ever calls `policy.forward()`
(the flow-matching loss), and `scripts/eval_adapter.py` does the same. Neither touches the
inference path at all: unnormalization, the KV-cache prefill, the forced-eager attention, or
the 10-step Euler integration in `sample_actions`. A good val loss and a broken sampler are
fully compatible. **The val losses in this file are unaffected and stand.**

Fixed, plus `_assert_unnormalizes()` in `eval_libero.py`: it pushes a zero action through the
post-processor and requires the dataset action mean back (pi0 normalizes actions MEAN_STD),
raising with expected-vs-actual otherwise. It tests the pipeline that runs rather than
asserting a config key exists.

Open-loop MAE on the same 16 val frames, before and after the one-line fix:

| | dx | dy | dz | rx | ry | rz | grip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| before | 0.534 | 0.462 | 0.473 | 0.462 | 0.509 | 0.843 | 0.241 |
| after | 0.169 | 0.095 | 0.220 | 0.022 | 0.048 | 0.036 | 0.163 |
| predict-the-mean baseline | 0.247 | 0.232 | 0.397 | 0.021 | 0.042 | 0.050 | 0.957 |

5/7 dims beat the baseline, up from 1/7. (`rx`/`ry` sit at the noise floor — rotation barely
varies in these demos, so there is almost nothing there to predict.) Closed loop on the same
checkpoint went 0/15 to 7/15.

**Generalise this:** any metric computed with `policy.forward()` certifies the forward pass
and nothing else. Before trusting a rollout harness, make it reproduce a policy already known
to be good. The removed `scripts/diagnose_libero.py` did that in four isolated checks
(open-loop actions, state construction, image orientation, action convention), each of which
can fail without the others noticing; rebuild it from git history if the harness changes.

### The dataset's own `meta/info.json` mislabels the state vector

It declares `observation.state` as `['x','y','z','rx','ry','rz','rw','gripper']` — position
plus a **quaternion** plus gripper. The statistics say otherwise: dim 3 has mean 3.06 and
range [2.497, 3.457], impossible for a quaternion component, and dims 6-7 are [0, 0.041] and
[-0.042, 0.001], the symmetric Panda finger-joint pair. The real layout is
`[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`, confirmed by check 2 of the diagnostic:
an env-built state matched a real dataset frame at **L2 0.0054** against a per-dim std of
0.10-0.41. The `names` field is inherited boilerplate from the port template. Trust the
stats, not the card.

The action convention was checked from the dataset alone (check 4, job 3108532; no
simulator involved): over one episode the recorded `action[:3]` correlates with the recorded
next-frame change in end-effector position at **+0.92 / +0.97 / +0.98** (dx/dy/dz), slopes
0.010 / 0.012 / 0.011, and the gripper action takes only the values -1 and 1. So the recorded
actions behave as normalized OSC_POSE position deltas, which is what `env.step(action)`
expects; a near-zero correlation would have meant absolute pose, joint space or a different
scaling. The slope sits at ~0.01, not the controller's `output_max` of 0.05, so do not use
0.05 as a unit conversion between action and metres.

### Installing the sim stack (the `egl-probe` wall)

**The recipe lives in README, "Simulator, for evaluation only".** It is the fresh-clone
document; this section is only the reasoning behind each step, and the commands are
deliberately not repeated here — they drifted once already.

`lerobot[libero]` pulls `hf-libero`, which pins `robomimic==0.2.0`, which depends on the
original `egl-probe` — and that package compiles C against EGL headers Athena's login node
does not have, so `uv pip install` dies at the CMake step. `hf-libero` ships `hf-egl-probe`
precisely to avoid this, but robomimic's own metadata still names the old one. `hf-egl-probe`
fails to build for the same reason.

Neither is actually needed: `egl_probe` is imported only inside `robomimic/envs/env_robosuite.py`
and `env_ig_momart.py`, and LIBERO's `OffScreenRenderEnv` wraps robosuite directly. Hence
`--no-deps` on `hf-libero`/`robomimic`, with robomimic's real dependencies re-added by hand.

This is why `lerobot[libero]` is still not in `pyproject.toml`: a plain `uv sync` cannot
express the `--no-deps` step and would fail at `egl-probe`.

`constraints.txt` pins torch/torchvision/torchcodec/transformers/numpy/lerobot/accelerate/
peft/huggingface-hub to their installed versions, so the install only adds; without it the
resolver will downgrade numpy or pull a PyPI (CPU) torch over the cu128 one. Verify with a
`uv pip freeze` diff before and after. The file is a `uv pip` flag only — `uv sync` and
`uv lock` have no `--constraint` and ignore it entirely.

Two sharp edges, both verified, both spelled out in README where the commands are:

- **`uv sync` deletes the sim stack** — it is exact by default and none of these packages are
  in `uv.lock`, so `--inexact` is mandatory afterwards.
- **`--reinstall-package opencv-python-headless` does not remove the GUI distribution** that
  `robosuite` pulls in, so an explicit `uv pip uninstall opencv-python` is required. The two
  distributions ship the same `cv2` module under different names, which is why no resolver
  reports the conflict. `scripts/verify_env.py` imports `cv2` *and* checks which
  distribution owns it: metadata alone reports a healthy cv2 in the window between the
  uninstall and the reinstall, when the shared files are gone.

### Runtime configuration

- `MUJOCO_GL=egl`. Compute nodes have `libEGL_nvidia.so.0` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`;
  login nodes have **no NVIDIA EGL vendor ICD**, and `libOSMesa.so.8` is absent so osmesa is
  not a fallback — anything touching the renderer must run under SLURM. (Precisely: a bare
  `import robosuite` on a login node dies with `'NoneType' object has no attribute
  'eglQueryString'`, raised in `OpenGL/raw/EGL/_types.py` from `mujoco/egl/egl_ext.py`. It *appears* to work under `slurm/common.sh` only because the
  micromamba FFmpeg prefix on `LD_LIBRARY_PATH` happens to ship a conda-forge `libEGL.so.1`;
  that is a loader accident, not a working renderer, which is why `verify_env.py` reads
  robosuite's version from metadata instead of importing it.)
- `MUJOCO_EGL_DEVICE_ID` / `EGL_DEVICE_ID` are set to `LOCAL_RANK` so each rank renders on
  its own GPU.
- `LIBERO_CONFIG_PATH=$SCRATCH/vla_finetuning/libero_config`. LIBERO writes a `config.yaml`
  on first import and asks for the dataset path **via an interactive `input()` at import
  time** when that file does not exist, which hangs a batch job until stdin EOFs (`EOFError`
  when stdin is closed). The trigger is the **missing file, not the unset variable**:
  `slurm/common.sh:49` exports `LIBERO_CONFIG_PATH` but never creates the config, so the
  file has to be written once, non-interactively (README has the heredoc). A guard that
  asserts only on the env var passes and then hangs.
- LIBERO's mesh/texture assets (409 MB on disk) are not in the wheel — the `hf-libero` wheel
  ships zero asset files. `get_assets_path()` (`libero/libero/__init__.py:68-97`) returns the
  package's own `assets/` directory **whenever that path merely exists**, and only otherwise
  downloads, to a hardcoded `os.path.join(os.path.expanduser("~"), ".cache", "libero",
  "assets")`. Neither `XDG_CACHE_HOME` nor `HF_HOME` moves that destination — it is a
  `snapshot_download(local_dir=…)` call — and there is no setting for it. Two consequences:
    - The default lands 409 MB in `$HOME`, whose 10 GiB quota blocks job submission at 90%
      (see below). The fix is to redirect the cache directory itself:
      `ln -sfn $VLA_ROOT/libero_assets ~/.cache/libero/assets`, then let LIBERO download
      through it. README has the recipe. This is preferred over planting a symlink inside
      site-packages: it leaves the package alone, so a venv rebuild costs nothing, and it
      avoids the trap below entirely.
    - The `exists()` check **fails open**: an `assets/` path inside the package that is
      empty, or a symlink to a still-empty target, permanently suppresses the download, and
      every object XML lookup then resolves into an empty tree with no error. If you do wire
      it up that way, populate the target *first*.
    - Current state on this account is the older wiring and still valid:
      `$SCRATCH/vla_finetuning/libero_assets`, symlinked from
      `<site-packages>/libero/libero/assets`, and named in `config.yaml`'s `assets:` key
      (which nothing actually reads — only `get_assets_path()` wires assets up).
      `verify_env.py` accepts either wiring, resolving them in the same order
      `get_assets_path()` does, and asserts five known asset subdirectories exist. It
      deliberately does **not** call `get_assets_path()`: that function is not a read-only
      probe, so a check built on it would itself trigger the 409 MB download into `$HOME`
      whenever the symlink is broken, then pass against the fresh copy and hide the breakage.

### Cost

~13.5 s per episode at 2 envs/rank; `--envs-per-rank 8` amortises the policy forward across
8 simultaneous rollouts. Episodes are dealt round-robin across ranks by `(task, trial)` so a
straggler rank's share is not task-specific.

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
$VLA_ROOT/            # = ./work in the repo by default; on Athena export it to
                      # $SCRATCH/vla_finetuning, which is what the numbers below assume
├── hf/               # HF_HOME — models + datasets
├── runs/             # VLA_RUNS — checkpoints (every entrypoint's --out default)
├── wandb/            # WANDB_DIR
├── .triton/          # TRITON_CACHE_DIR
├── libero_config/    # LIBERO_CONFIG_PATH
├── ffmpeg/           # VLA_FFMPEG_PREFIX — micromamba prefix, see below
└── venvs/vla/        # VLA_VENV / UV_PROJECT_ENVIRONMENT
```

One file resolves all of it: **`slurm/common.sh`**, which every `slurm/*.sbatch` sources as
its first statement (`source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"`). It sources `.env`,
defaults every variable above off `$VLA_ROOT`, puts FFmpeg on `LD_LIBRARY_PATH`, and
activates the venv — so no job script contains a hardcoded path, and a different site or a
different filesystem only needs `export VLA_ROOT=...`. Each value is only *defaulted*, so
an exported one wins.

Outside a job, do exactly what a job does:

```bash
source slurm/common.sh
```

`src/paths.py` mirrors the same resolution for Python (`run_dir()`, `ffmpeg_prefix()`), so
an entrypoint launched with nothing exported still writes to the same place.

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
export VLA_ROOT=$SCRATCH/vla_finetuning           # default is ./work, in the 10 GB home
export UV_PROJECT_ENVIRONMENT=$VLA_ROOT/venvs/vla
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
     --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train_pi0.py
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
- **Phase separation** (per `PROJECT.md`): Phase 1 was a raw `torch.distributed` loop on a
  toy model, there to make the communication layer legible; it was removed once the repo was
  cut down to what produced the published checkpoints. `train_pi0.py` keeps that raw-DDP
  structure with the real model; Phase 2B moves to Accelerate + FSDP.
- Barrier before teardown and call `destroy_process_group()`, or jobs hang at exit and waste
  allocation.
- Log effective batch size explicitly (per-device × grad accumulation × world size); it is
  the number that makes W&B curves comparable across runs.

## Where Phase 2 goes

Add `configs/` for accelerate/FSDP/DeepSpeed yaml and hyperparams; keep `slurm/` one
`.sbatch` per experiment shape. `logs/`, checkpoints, venvs, and datasets are gitignored —
none of it belongs in a 10 GB home directory or in the published repo.

## Next steps

**A and B are both done** (see "Closed-loop LIBERO evaluation" and "Full fine-tuning under
FSDP" above). **`TODO.md` holds the full outstanding list** — publishing prerequisites, the
remaining evaluation gaps and housekeeping — and is the file to read first when picking this
back up.

### A. Simulator rollouts — **done**, see "Closed-loop LIBERO evaluation" above

`scripts/eval_libero.py` (+ `render_results.py`) and `slurm/pi0_libero_eval.sbatch`; the
debugging tools that found the post-processor bug were removed in the cleanup. Measured:
base 0.0%, LoRA 54.2% +- 2.2, full FT 56.6% +- 2.2 over 500 episodes each.
The original planning notes below proved accurate about the time sink — EGL and the install,
not the policy code — and missed the one that actually cost a run: the post-processor.

#### Original planning notes

*(Kept as written before the work, for the record. The gap described here is now closed.)*

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

### B. Full fine-tuning under FSDP — **done**, see the section above

`train_pi0_fsdp.py` plus `pi0_fsdp_bench.sbatch` and `pi0_full_train.sbatch` (the preflight
and smoke jobs were removed in the cleanup). Two notes carried forward from planning:

**There is no `configs/` directory, deliberately.** The sharding strategy is a CLI flag
(`--sharding`) and the FSDP plugin is built in Python, because the auto-wrap policy pi0
requires cannot be expressed in an accelerate YAML — it is a callable that matches
`GemmaMLP`/`SiglipEncoderLayer` rather than the decoder-layer class YAML would name. Shipping
YAML that the code then overrides would be two sources of truth for the one setting the whole
experiment varies, and the reader could not tell which one ran. Launch with `torchrun`, like
everything else in the repo.

**Do NOT make full FT the production recipe.** Full FT of 4B params on 392 episodes invites
catastrophic forgetting of pi0's pretraining, and it breaks deliverable #2 (a multi-GB model
instead of an 11 MB adapter). LoRA + DDP (`train_pi0.py`, `slurm/pi0_train.sbatch`) stays the
production run; `slurm/pi0_full_train.sbatch` exists so the comparison is a number rather
than an assumption.

### Also outstanding, smaller

- **Push `adapter_best/` and `model_best/` to the Hub** (deliverable #2). Cards are written
  (`model_cards/`) and `scripts/publish_to_hub.py` applies them over PEFT's regenerated stub.
  Both quote the re-measured val loss with its std *and* the success rate, and both warn
  against reading our figures against published pi0 LIBERO tables.
- **Out-of-distribution evaluation is now the interesting open question.** `eval_libero.py
  --suite libero_object|libero_goal|libero_10` needs no retraining and ~1 GPU-hour. LoRA and
  full FT tie in distribution but diverge sharply on individual tasks, so whether they
  diverge further out of distribution is a real question, not a formality. Leave `--dataset`
  on the training set so normalization stats stay deployment-faithful.
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
