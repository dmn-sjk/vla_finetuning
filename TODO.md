# TODO

State as of 2026-08-29. Phases 1, 2 (LoRA+DDP) and 2B (full FT under FSDP) are implemented
and measured; `CLAUDE.md` has the engineering notes and every measured number. What follows
is what is *not* done.

Ordered by what blocks what. Items 1–2 block everything else.

---

## 0. Where the checkpoints live

Moved off `$SCRATCH` (purgeable) on 2026-08-29, verified with an `rsync --checksum` pass
before the source was deleted:

```
$PLG_GROUPS_STORAGE/plgguseby/dmns/vla/
├── pi0_libero/   243 MB   LoRA production run (job 2988297)
│   ├── adapter_best/      48 MB — THE deliverable, val 0.3344 ± 0.0020
│   ├── adapter/           48 MB — final epoch, val 0.3398. Do not publish.
│   └── best.pt, last.pt  144 MB each — resume state
└── pi0_full/      36 GB   full-FT run (job 2991036)
    ├── model_best/       7.6 GB — epoch 0, val 0.3350 ± 0.0014
    ├── model/            7.6 GB — epoch 4, val 0.8361. Overfitting exhibit only.
    └── state/             45 GB apparent — FSDP resume checkpoint
```

Reference: base `lerobot/pi0` scores **0.8719 ± 0.0048** on the same 40 episodes. Both
*best* checkpoints are far better than base; both *final* checkpoints are worse than their
own best, and full FT's final has given back essentially the whole gain.

---

## 1. BLOCKER: `$HOME` quota — no jobs can be submitted

`$HOME` is at **9.34 / 10 GiB (93.4%)** and Athena's job-submit plugin rejects outright:

```
sbatch: error: Storage quota of $HOME exceeded (over 90% full)!
```

Confirmed with `sbatch --test-only`. Nothing below that needs a GPU can start until this is
cleared. The growth is **not** from this project — largest consumers are `.local/share`
(2.8 G), `.vscode-server` (2.8 G), `.cache/torch` (1.1 G), `.cursor-server` (353 M).
Clearing `.cache/torch` alone would probably suffice.

## 2. BLOCKER: Phase 2B is uncommitted

Seven files sit in the working tree with no commit:

```
 M CLAUDE.md  README.md  scripts/eval_adapter.py
?? train_pi0_fsdp.py
?? slurm/pi0_fsdp_{preflight,smoke,bench,verify_export}.sbatch
?? slurm/pi0_{full_train,compare_checkpoints}.sbatch
```

`train_pi0.py` is deliberately byte-identical to the committed version — the LoRA + DDP
production recipe was never touched. Note `scripts/eval_adapter.py` carries a **bug fix**
(see §5) that changes what `--repeats` reports, so anything measured with it before
2026-08-18 has a meaningless spread.

---

## 3. Evaluation: there is no test set, and loss is the wrong metric anyway

Two separate problems, often conflated.

### 3a. No test split exists

The 432 episodes of `lerobot/libero_spatial_image` are split **392 train / 40 val**, with
nothing held back. Worse, those 40 val episodes were used for best-epoch selection in both
runs, so they are not a clean set to *report* on either — the reported numbers are
selection-biased, if only slightly.

Options, in increasing order of cost:

- **Evaluate on a different LIBERO suite** (`libero_object`, `libero_goal`, `libero_10`) as
  an out-of-distribution held-out set. Cheapest, needs no retraining, and answers a more
  interesting question than an in-distribution test loss: did adapting to LIBERO-spatial
  cost anything elsewhere? Expect the full-FT `model_best` to look much worse than the LoRA
  adapter here — that is the prediction worth testing.
- **Re-split 392/20/20** and retrain both runs. Clean, but invalidates every number
  currently in `CLAUDE.md` and costs ~4 GPU-hours (LoRA ~1 h on 8 GPUs, full FT ~3.3 h on 4).

Whichever is chosen, run it through `scripts/eval_adapter.py` with `--repeats 5` so the
number carries a spread. Do not quote a bare val loss (see §5).

### 3b. Success rates, not loss, are what the literature reports

π0 and OpenVLA report LIBERO **task success rates** from closed-loop rollouts. A
flow-matching validation loss does not convert to that, so **no test loss, however clean,
makes these results comparable to published tables.** This is the single biggest credibility
gap in the project.

Needs:
- `lerobot[libero]` (adds `hf-libero`, pulls LIBERO sim + MuJoCo) — deliberately excluded
  from `pyproject.toml` today; re-lock and re-verify when adding.
- Headless GL on compute nodes: expect `MUJOCO_GL=egl` (or `osmesa`). This, not the policy
  code, is the likely time sink.
- Rollouts use `policy.select_action`, not `forward`; `PI0Config.n_action_steps` controls
  how much of the 50-step chunk is executed per inference.
- Rollouts are sequential per episode and CPU/sim-bound — parallelise across *episodes* by
  rank, not by batching.

---

## 4. Publishing to the HF Hub

Account `dmnsjk` currently has **0 models**. Deliverable #2 is entirely outstanding.

### 4a. Decide what to publish

- **`adapter_best/` — yes.** 48 MB, val 0.3344 ± 0.0020, and the artefact the project is
  actually about.
- **`model_best/` — probably not as a *model*.** 7.6 GB for a result statistically tied with
  the 48 MB adapter (0.3350 ± 0.0014 vs 0.3344 ± 0.0020; the 0.0006 gap is inside the noise).
  It is an artefact of the sharding experiment, not a better policy. If published at all, the
  card must say so plainly.
- **`adapter/` and `model/` (final-epoch) — no.** Both are worse than their own best, and
  `model/` has lost essentially the entire fine-tuning gain (0.8361 vs base 0.8719).

### 4b. Write a real model card

`adapter_best/README.md` is PEFT's auto-generated stub — every field reads
`[More Information Needed]`. It is also regenerated on every export, so the card belongs in
the repo and should be copied in at publish time, not hand-edited in the run directory.

Must contain: base model and link; dataset and the exact episode split; LoRA config
(r=32, alpha=64, dropout=0.05, `attention` target preset, 12.65M trainable = 0.31%);
optimizer and schedule; hardware (8× A100-40GB, 67 min, 89.3 samples/s, 11.7 GiB peak);
**val loss with its std and the 0.8719 base reference**; an inference snippet; the licence
note in §4c; and an explicit statement that **no task success rate has been measured**.

### 4c. Licence — not a blocker, but must be documented

Checked on 2026-08-29:

- `lerobot/pi0` is **apache-2.0** and **not gated** → redistributing derivatives is fine.
- `google/paligemma-3b-pt-224` is **gemma licence, gated (manual approval)**. π0 loads its
  *processor/tokenizer* from there, so anyone using our checkpoint must accept that licence
  first. Say so in the card, or the first thing users hit is a 403.

### 4d. Mechanics

`HF_TOKEN` in `.env` has write scope (verified). Push with `huggingface_hub`; LFS is handled
automatically. Suggested repo names: `dmnsjk/pi0-libero-spatial-lora` and, if published,
`dmnsjk/pi0-libero-spatial-full`.

---

## 5. Carry-forward warnings

- **`scripts/eval_adapter.py --repeats` was reporting a structural zero** until 2026-08-18.
  The script seeded with `torch.manual_seed()` before calling `T.evaluate()`, which re-seeds
  internally with its own default and overwrote it — every repeat evaluated identically, so
  `std`/`spread` were always exactly 0.0000. Fixed by passing `seed=` through. The historical
  `0.3343 ± 0.0015` predates the internal seeding and stands; anything measured with
  `--repeats` between those two changes has a meaningless spread. **Re-measure before
  quoting.**
- **Validation loss is a stochastic estimate.** π0 is flow-matching: every forward samples a
  diffusion timestep and noise. Never report a bare val loss; always `--repeats 5`.
- **`train_pi0.py`'s published samples/s is not comparable to `train_pi0_fsdp.py`'s.** The
  former's throughput window includes the trailing validation pass; the latter times training
  segments only.

---

## 6. Housekeeping

- **~137 GB of smoke/benchmark debris still on `$SCRATCH`**, none of it of any value now:
  `pi0_fsdp_smoke` (45 G), `pi0_fsdp_smoke-zero2` (60 G), `pi0_fsdp_pre-lora` (16 G),
  `pi0_fsdp_smoke-lora` (16 G), plus small older dirs (`pi0_smoke`, `pi0_scaling`,
  `pi0_libero_run1_overfit`, `smoke`, `smoke2`, `cpucheck`).
- **`pi0_full/state/` is 45 GB of FSDP resume state** for a run that finished all 5 epochs
  and is past its minimum. Nothing will be resumed from it. Delete unless the sharded-
  checkpoint layout is wanted as an exhibit.
- **W&B report** (deliverable #3) is not assembled; runs are logged but no public report
  exists.
- The 8-GPU benchmark row **`8 GPUs | DDP | full FT`** is recorded as `ROW_FAILED` rather
  than `OOM_EXPECTED`, because `--oom-ok` is only safe at world size 1. The OOM is visible in
  the traceback in `logs/pi0-fsdp-bench-2990653.out`; if a cleaner table row is wanted, that
  needs a multi-rank-safe OOM handler.
