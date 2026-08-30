# TODO

State as of 2026-08-30. Phases 1, 2 (LoRA+DDP) and 2B (full FT under FSDP) are implemented
and measured; `CLAUDE.md` has the engineering notes and every measured number. What follows
is what is *not* done.

Nothing is blocked. What remains is publishing, one optional evaluation, and cleanup.

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

## 1-2. Former blockers — both cleared

`$HOME` quota is back to 6.5 / 10 GiB and jobs submit normally. Phase 2B was committed
(`bb39235`). Keep LIBERO's assets out of `$HOME`: they download to `~/.cache/libero` by
default and are now redirected to `$SCRATCH` (see `CLAUDE.md`, "Runtime configuration").

---

## 3. Evaluation

**Performance improvements have their own file: `IMPROVEMENTS.md`** — what the ~42-point gap
to published pi0 LIBERO numbers plausibly is, what each fix costs, and in what order. Its
first item (~30 min) decides whether the rest is worth doing at all.


### 3b. Success rates — **DONE** (job 3108565, 2026-08-30)

500 episodes per checkpoint on the standard LIBERO protocol. base 0.0% (0/500),
LoRA `adapter_best` **54.2% ± 2.2** (271/500), full FT `model_best` **56.6% ± 2.2**
(283/500). The +2.4 point gap has SE 3.1 (z = 0.76): **not significant**, independently
confirming the validation-loss tie. Details and caveats in `CLAUDE.md`, "Success rates,
measured".

The path there cost one wrong run and is worth not repeating: the post-processor was never
given the dataset's stats, so rollouts drove the arm with z-scores and a good checkpoint
scored 0/15. See `CLAUDE.md`, "A good validation loss does not mean the policy works".
`scripts/diagnose_libero.py` is the tool that localised it.

### 3a. No test split exists — still true, and still mostly fine

The 432 episodes are split 392 train / 40 val with nothing held back, and those 40 were also
used for best-epoch selection, so the *validation loss* is mildly selection-biased. Two
things reduce how much this matters now:

- The **success rates are not affected at all.** They are measured on the simulator's own 50
  saved initial states per task, which are not dataset episodes and played no part in
  checkpoint selection. The headline metric is clean.
- The bias is small where it applies. `adapter_best` scored 0.3273 at selection time and
  0.3344 on a clean 3-seed reload — a ~0.007 gap, matching the documented training-time
  noise, against a 0.54 improvement over base. **Publish the re-measured 0.3344, never the
  0.3273 that drove selection.** For full FT the best epoch was epoch 0 of a monotonically
  worsening curve, so there was effectively nothing to cherry-pick.

Re-splitting and retraining to fix ~0.007 of bias on a metric that is no longer the headline
is not worth ~4 GPU-hours. **The genuinely open evaluation question is out-of-distribution
generalization**, not a clean in-distribution test loss:

- **Evaluate on another LIBERO suite** (`libero_object`, `libero_goal`, `libero_10`; all
  available as `_image` ports). No retraining, ~1 GPU-hour with `eval_libero.py --suite`.
  The interesting prediction to test: LoRA and full FT tie in-distribution but differ sharply
  per task (task 5: 30% vs 6%; task 9: 34% vs 62%), so they may diverge much further out of
  distribution — full FT touched all 4.03B parameters and has more to forget.
  `--dataset` already defaults to the *training* set's normalization stats, which is the
  correct deployment-faithful choice; do not point it at the target suite.

## 4. Publishing to the HF Hub

Account `dmnsjk` currently has **0 models**. Deliverable #2 is entirely outstanding.

### 4a. Decide what to publish

- **`adapter_best/` — yes.** 48 MB, val 0.3344 ± 0.0020, and the artefact the project is
  actually about.
- **`model_best/` — yes, but framed as the control arm.** 7.6 GB for a result tied with the
  48 MB adapter on *both* metrics (56.6% ± 2.2 vs 54.2% ± 2.2, z = 0.76; 0.3350 ± 0.0014 vs
  0.3344 ± 0.0020). It is not a better policy. It is now worth publishing anyway because the
  per-task breakdown shows the two are genuinely *different* policies that happen to tie,
  which makes the pair more useful than either alone. The card says all of this plainly.
- **`adapter/` and `model/` (final-epoch) — no.** Both are worse than their own best, and
  `model/` has lost essentially the entire fine-tuning gain (0.8361 vs base 0.8719).

### 4b. Write a real model card

`adapter_best/README.md` is PEFT's auto-generated stub — every field reads
`[More Information Needed]`. It is also regenerated on every export, so the card belongs in
the repo and should be copied in at publish time, not hand-edited in the run directory.

**Written**: `model_cards/pi0-libero-spatial-{lora,full}.md`, applied at publish time by
`scripts/publish_to_hub.py` (which overwrites whatever stub PEFT regenerated). Both carry
base model and link, dataset and exact split, LoRA config, optimizer/schedule, hardware and
wall clock, the success-rate table with per-task breakdown, the re-measured val loss with its
std and the 0.8719 base reference, an inference snippet, the `rot180` warning, and the §4c
licence note. Both state explicitly that 54.2%/56.6% must **not** be read against the ~96%
published for pi0 on LIBERO-spatial, and that the 0.0% base row is a zero-shot floor rather
than the paper's number.

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

- **A validation loss certifies `policy.forward()` and nothing else.** It never touches the
  inference path — unnormalization, KV-cache prefill, the Euler integration in
  `sample_actions`. A good val loss is fully compatible with a policy that scores 0% in
  closed loop; that is exactly what happened here. Before trusting any rollout harness, make
  it reproduce a checkpoint already known to be good.
- **`make_pre_post_processors` needs `postprocessor_overrides` too.** Passing only
  `preprocessor_overrides` leaves the unnormalizer on the pretrained checkpoint's stats and
  the policy emits z-scores. `_assert_unnormalizes()` in `eval_libero.py` now catches this at
  load time; any new script that builds processors should call it or repeat the check.
- **LIBERO camera frames need `rot180`** to match the training data. Wrong orientation runs
  silently and scores ~0%. Re-run `eval_libero.py --verify-obs` if the dataset changes.
- **Do not trust `meta/info.json`'s `names` for `observation.state`.** It claims a quaternion;
  the real layout is `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`. Trust the stats.

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
