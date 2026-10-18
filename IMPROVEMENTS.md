# Closing the gap to published π0 LIBERO numbers

State as of 2026-08-30. We measure **54.2% ± 2.2** on LIBERO-spatial (LoRA `adapter_best`) and
**56.6% ± 2.2** (full FT `model_best`). openpi reports **96.8–97.4%** for π0 on the same suite.
This file is about the ~42-point gap: what plausibly causes it, what each fix costs, and in
what order to attempt them.

Everything labelled *measured* comes from this repo's own runs (see `docs/notes.md`). Everything
labelled *estimated* is arithmetic on those measurements. Expected gains are **guesses** unless
stated otherwise — that is exactly why item 1 comes first.

---

## 0. What is already ruled out

- **It is not LoRA.** Full fine-tuning of all 4.03B parameters scored 56.6% against LoRA's
  54.2% — a +2.4 point difference with SE 3.1 (z = 0.76), and the validation losses tie too
  (0.3350 vs 0.3344, inside ±0.002 seed noise). Parameter efficiency is not costing us 42
  points. Spending compute on "full fine-tune instead" is spending it on a measured null.
- **It is not the evaluation harness being broken.** It was, once — see `docs/notes.md`, "LIBERO
  evaluation" — and that is fixed and guarded. The
  protocol matches the standard convention (10 tasks × 50 trials from the suite's saved
  initial states, 280-step cap, which is *more* generous than OpenVLA's 220).
- **It is not more GPUs.** Full FT OOMs under DDP even on 8× A100-40GB; only sharding makes it
  run at all. Hardware is not the binding constraint on quality.

---

## 1. FIRST: find out whose gap it is  ⏱ ~30 min

**Do this before spending hours on anything else.** It splits the 42 points into "our recipe"
and "the lerobot stack", and those two lead to completely different work.

`lerobot/pi0_libero_finetuned_v044` is lerobot's **own** π0 fine-tuned on LIBERO, in exactly
our checkpoint layout (`config.json` + `model.safetensors` + processor configs, and it ships
its own normalizer stats). Run it through `scripts/eval_libero.py` unchanged.

| outcome | interpretation | what to do next |
| --- | --- | --- |
| ~97% | our harness is sound; the whole gap is **our training recipe** | items 3–6 below, expect real gains |
| ~65–70% | the gap is largely **the lerobot stack**, not us | items 3–6 buy far less; consider reporting upstream and re-scoping the claim |
| ~54% | the gap is the stack almost entirely | our recipe is already near the stack's ceiling |

The middle row is the one to expect. [lerobot issue #2114](https://github.com/huggingface/lerobot/issues/2114)
reports a full π0 fine-tune through lerobot reaching **69% on spatial** against the published
~97%, and there are unresolved companion issues
([#952](https://github.com/huggingface/lerobot/issues/952),
[openpi #804](https://github.com/Physical-Intelligence/openpi/issues/804)). If that is right,
roughly 28 of our 42 points are not ours to recover.

**Two small code changes are needed first**, because this checkpoint is not ours:

- `--checkpoint-stats`: skip the `--dataset` normalization override and use the stats the
  checkpoint shipped. It was trained on the combined LIBERO mix, so forcing spatial's stats
  on it would misscale its actions — the same class of bug as the one already fixed.
- `--camera-keys`: it was trained on `HuggingFaceVLA/libero`, which names its cameras
  `observation.images.image` / **`image2`**, not `image` / **`wrist_image`** as
  `lerobot/libero_spatial_image` does. Wrong keys here fail silently.

---

## 2. Free: tune what we never tuned  ⏱ ~40 min, no training

`n_action_steps=5` follows openpi's LIBERO replan interval. It was never tested against
alternatives on *our* checkpoint, and it is a pure inference-time knob: sweep
`{1, 5, 10, 25, 50}` at 100 episodes each (~8 min per setting on 4 GPUs).

Worth doing because the two published-checkpoint configs disagree: openpi replans every 5
steps, while `lerobot/pi0_libero_finetuned_v044`'s own config sets `n_action_steps: 50`
(execute the whole chunk open-loop). One of those is much better for this model and we do not
know which.

Report the sweep honestly — picking the best of five settings on the same episodes is
selection on noise unless the winner is re-measured on a fresh 500-episode run.

---

## 3. Train on all four LIBERO suites  ⏱ 1–5.4 h

**The single biggest lever, and the one our own data most directly supports.** Our full FT run
hit its best validation loss at **epoch 0** and then rose monotonically to base-model quality
by epoch 4 (0.3344 → 0.8361 while train loss fell to 0.084). That is not a tuning failure, it
is 392 episodes being exhausted. openpi trains 30k steps precisely *because* it has 4.3× more
data; 30k steps on spatial alone would collapse.

**`HuggingFaceVLA/libero` is the combined dataset, already packaged** — 1,693 episodes /
273,465 frames / 32.5 GiB, exactly the sum of the four suites. No `MultiLeRobotDataset`
plumbing and no per-suite normalization merging required. **But its camera keys are
`image`/`image2`, not `image`/`wrist_image`**, so training on it produces a policy with a
different observation contract than our current checkpoints — `eval_libero.py` needs the same
`--camera-keys` flag as item 1.

### Wall-clock, from our measured throughputs

Holding out ~160 episodes (~10%, same spirit as the current 40/432) leaves ~247,600 train
frames. Throughputs are *measured*: LoRA+DDP 89.3 samples/s on 8 GPUs, full FT FSDP 49.7 on 8
and 19.2 on 4.

| training budget | LoRA, 8 GPU | full FT, 8 GPU | full FT, 4 GPU | epochs |
| --- | --- | --- | --- | --- |
| lerobot v044 recipe (10k × 32 = 320k samples) | **1.0 h** | 1.8 h | 4.6 h | 1.3 |
| openpi recipe (30k × 32 = 960k samples) | **3.0 h** | 5.4 h | 13.9 h | 3.9 |
| 7 epochs, as our spatial run | 5.4 h | 9.7 h | 25.1 h | 7.0 |

**Recommended: LoRA on 8 GPUs at the openpi sample budget — ~3 h**, inside a single 8-hour
allocation with wide margin.

Three caveats on those numbers:

- **They assume our measured per-device batch.** 89.3 samples/s was measured at batch 16 per
  device. Running a *true* effective batch of 32 means 4 per device on 8 GPUs, which will not
  sustain that rate — expect 40–60% of it, so ~5–7 h rather than 3 h. If step count matters
  less than sample count, use 8 per device with no accumulation (effective 64, 15k steps) and
  keep near-full throughput.
- **Dataloader risk.** 273k PNG frames on Lustre with random access, versus 48k today. If the
  loader cannot feed 89 samples/s the whole table stretches. Measure on the first epoch before
  trusting the estimate.
- **Add ~20 min** for the 32.5 GiB download (`$SCRATCH` has ~100 TB free; the HF cache is
  currently 63 GB).

### Then re-evaluating costs more than before

4 suites × 500 episodes = 2,000 episodes per checkpoint, at the measured 23–27 min per 500 on
4 GPUs → **1.5–1.8 h per checkpoint**. Budget that separately from training.

---

## 4. Recipe deltas we can copy verbatim  ⏱ free (config only)

`lerobot/pi0_libero_finetuned_v044`'s `train_config.json` is public. Ours differs in five
concrete ways, none of which we chose deliberately:

| | lerobot v044 | openpi | **our LoRA run** | **our full-FT run** |
| --- | --- | --- | --- | --- |
| optimizer steps | 10,000 (+ continued from a prior run) | 30,000 | **1,314** | **~188** (best was epoch 0) |
| batch size | 32 | 32 | **256** effective | **256** effective |
| weight decay | 0.01 | — | **0.0** | 0.01 ✓ |
| Adam betas | (0.9, 0.95) | — | **(0.9, 0.999)** default | (0.9, 0.95) ✓ |
| peak LR | 2.5e-5 | — | 1e-4 | 2.5e-5 ✓ |
| LR warmup | **8,000 steps**, cosine to 2.5e-6 | — | 200 steps | **50 steps** |
| EMA | not in config | **0.999** | none | none |
| camera slots | 3 (2 real + 1 empty) | 3 | **2** | **2** |

**Weight decay, betas and peak LR are already ruled out.** `train_pi0_fsdp.py` defaults to
exactly the reference values (`--weight-decay 0.01`, `--betas 0.9 0.95`, `--lr 2.5e-5`), so
the full-FT run had all three "right" and still landed at 56.6% — statistically tied
(z = 0.76) with the LoRA run that had all three "wrong". Whatever is costing us 40 points, it
is not these. Adopt them for consistency if convenient, but do not expect points.

*(Do not read the `optimizer_*` fields in our checkpoints' `config.json` as evidence either
way — those are lerobot preset defaults that get serialized regardless; both trainers build
their own AdamW.)*

### We train with 2 camera slots; the reference setups use 3

Found on 2026-08-30 while preparing item 1, and **not a deliberate choice on our part**.

`lerobot/pi0` declares three camera inputs (`camera0/1/2`), and openpi's LIBERO setup feeds
three image slots — the two real LIBERO cameras plus one empty padding camera, which is why
`lerobot/pi0_libero_finetuned_v044` carries `empty_cameras: 1` and an
`observation.images.empty_camera_0` feature. **Our runs use `empty_cameras: 0` and two
slots**, because `PI0Config(pretrained_path=...)` is a plain dataclass constructor that never
reads the checkpoint's `config.json`: features get filled from the dataset and structural
fields keep library defaults. We inherited a default rather than picking a value.

π0 pads a missing image feature with -1 and a zero attention mask, so two slots *works* — but
it changes the prefix token layout relative to how the base model was pretrained, and relative
to both reference recipes. Whether that costs accuracy is untested.

Cheap to test properly: retrain with `empty_cameras=1`. Note that evaluating our *existing*
checkpoints with 3 slots would not test this — it would just introduce a train/test mismatch
they don't have. Fold it into item 3's run rather than spending a separate one.

The recipe items that actually stand out, then:

- **Effective batch 256 vs 32.** At the same sample budget this costs us **8× fewer optimizer
  steps**. Combined with 4.3× less data, our LoRA run took 1,314 steps against openpi's
  30,000 — **23× fewer**. This is free to change and should be changed alongside item 3.
- **Warmup of 8,000 steps** is longer than our entire training run. That only makes sense at
  their step counts, but it says our 50-step warmup is not a like-for-like choice.

(`weight_decay` and `betas` are covered above — measured not to matter.)

---

## 5. EMA of the weights  ⏱ ~1 h to implement

openpi uses **EMA with decay 0.999**; we use none. EMA is well established for
flow-matching/diffusion policies and is a plausible contributor.

Cost asymmetry matters here:

- **LoRA: nearly free.** One extra copy of 12.65M trainable parameters (~50 MB fp32).
- **Full FT: expensive.** A shadow copy of 4.03B parameters is ~16 GB fp32, on top of a run
  that already needs 4 GPUs minimum and peaks at 25.1 GiB. Would need sharding of the EMA
  buffer too, or offload to CPU.

Given LoRA and full FT are measured to tie, **implement it for the LoRA path only.**

---

## 6. Capacity and regularization  ⏱ folded into item 3

Only worth touching *after* item 3, because on 392 episodes more capacity means faster
overfitting.

- **LoRA capacity.** `--lora-target attention-mlp` is already implemented (53M params, 1.3%,
  vs 12.65M/0.31% today). With 4.3× the data this becomes reasonable rather than reckless.
- **Image augmentation.** lerobot's config carries a ColorJitter/brightness/contrast/saturation
  transform block but ships it **disabled** (`"enable": false`). We do not use it either.
  Cheap to switch on; unknown value.
- **`--weight-decay` is 0.0** today (see item 4).

---

## Suggested order

| # | action | cost | gates |
| --- | --- | --- | --- |
| 1 | evaluate `lerobot/pi0_libero_finetuned_v044` in our harness | ~30 min | decides whether 3–6 are worth doing |
| 2 | `n_action_steps` sweep on our checkpoint | ~40 min | free; independent of 1 |
| 3 | LoRA on `HuggingFaceVLA/libero`, openpi sample budget, batch 64, weight decay 0.01, **`empty_cameras=1`** | ~3–5 h | needs 1's verdict |
| 4 | re-evaluate on all four suites | ~1.5–1.8 h | after 3 |
| 5 | add EMA (LoRA path), retrain | ~3–5 h | only if 3 helped |

Items 1 and 2 together are **~70 minutes** and between them determine whether the remaining
~10 GPU-hours are worth spending. Do not skip them.

## What none of this fixes

If item 1 comes back at ~69%, then a substantial part of the gap lives in the lerobot π0 port
rather than in anything this repo controls, and no amount of retraining here reaches 97%. In
that case the honest move is what the model cards already do: state the protocol precisely,
report the numbers as measured, and decline the comparison to openpi's table rather than
implying our pipeline is equivalent to theirs.
