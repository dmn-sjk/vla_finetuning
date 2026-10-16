---
base_model: lerobot/pi0
library_name: peft
license: apache-2.0
pipeline_tag: robotics
tags:
  - lerobot
  - robotics
  - vision-language-action
  - pi0
  - lora
  - libero
datasets:
  - lerobot/libero_spatial_image
---

# pi0-libero-spatial-lora

A LoRA adapter that fine-tunes [`lerobot/pi0`](https://huggingface.co/lerobot/pi0) on the
**LIBERO-Spatial** manipulation suite. 12.65M trainable parameters (0.31% of the 4.03B
base), a 49 MB artefact, trained in 67 minutes on 8x A100-40GB with plain DDP.

## Model details

| | |
| --- | --- |
| Base model | `lerobot/pi0` (PaliGemma-3B backbone + flow-matching action expert, 4.03B params) |
| Method | LoRA (PEFT), r=32, alpha=64, dropout=0.05 |
| Adapted modules | q/k/v/o projections of **both** towers (VLM backbone and action expert), plus the state/action projections `state_proj`, `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out` |
| Trainable params | 12,648,448 (0.31%) |
| Dataset | [`lerobot/libero_spatial_image`](https://huggingface.co/datasets/lerobot/libero_spatial_image), 432 episodes / 52,970 frames, 10 tasks |
| Split | 392 train / 40 validation, held out **by episode** (a frame-level split leaks: consecutive frames of a trajectory are near-duplicates) |
| Action space | 7-D end-effector delta + gripper, 50-step chunk at 10 Hz |
| Observations | `observation.images.image` (agentview 256x256), `observation.images.wrist_image` (eye-in-hand 256x256), `observation.state` (8-D: eef position, eef axis-angle, 2 gripper joint positions) |

## Training

| | |
| --- | --- |
| Hardware | 8x NVIDIA A100-40GB (single node), DDP |
| Wall clock | 67 minutes, 89.3 samples/s, 11.7 GiB peak per GPU |
| Epochs | 7 (checkpoint selected by best validation loss) |
| Effective batch | 256 (16 per device x 2 accumulation x 8 ranks) |
| Optimizer | AdamW, lr 1e-4, weight decay 0.0, cosine schedule |
| Precision | bf16 autocast, fp32 master weights |
| Other | gradient checkpointing on, seed 42 |


## Results

Two independent measurements, both on held-out data.

### Closed-loop success rate

500 episodes: all 10 LIBERO-Spatial tasks x 50 trials, each trial starting from the suite's
own *i*-th saved initial state, 280-step cap, `n_action_steps=5`. This is the standard
LIBERO/OpenVLA evaluation protocol.

| checkpoint | success rate ± SE | successes |
| --- | --- | --- |
| base `lerobot/pi0` (zero-shot) | 0.0% ± 0.0 | 0/500 |
| **LoRA `adapter_best` (this model)** | **54.2% ± 2.2** | 271/500 |
| [full fine-tune `model_best`](https://huggingface.co/dmnsjk/pi0-libero-spatial-full) | 56.6% ± 2.2 | 283/500 |

± is the **binomial standard error** of each rate, √(p(1−p)/n) with n = 500. It measures evaluation noise for this one checkpoint (how much the rate would move on a re-run). For the 0/500 zero-shot row it is 0 by construction. A 95% Wilson interval puts the true rate between 0% and 0.8%.

Per task:

| task | instruction | this adapter | full FT |
| --- | --- | --- | --- |
| 0 | between the plate and the ramekin | 64% | 62% |
| 1 | next to the ramekin | 56% | 42% |
| 2 | from table center | 68% | 76% |
| 3 | on the cookie box | 58% | 60% |
| 4 | in the top drawer of the wooden cabinet | 38% | 44% |
| 5 | on the ramekin | 30% | 6% |
| 6 | next to the cookie box | 72% | 76% |
| 7 | on the stove | 56% | 70% |
| 8 | next to the plate | 66% | 68% |
| 9 | on the wooden cabinet | 34% | 62% |

Each per-task rate is from 50 episodes, so it carries a binomial standard error of up to ±7 points. Differences of a few points between tasks or checkpoints are noise.

### Validation loss

Flow-matching loss on the 40 held-out episodes, re-measured by reloading each artefact fresh and averaging 3 seeded passes (pi0 samples a diffusion timestep and noise on every forward, so a single number is a stochastic estimate):

| checkpoint | val loss (mean ± std, n=3) | trainable params | artefact | min 40GB GPUs |
| --- | --- | --- | --- | --- |
| base `lerobot/pi0` (zero-shot) | 0.8719 ± 0.0048 | — | — | — |
| **LoRA `adapter_best` (this model)** | **0.3344 ± 0.0020** | 12.65M | 49 MB | 1 |
| [full fine-tune `model_best`](https://huggingface.co/dmnsjk/pi0-libero-spatial-full) | 0.3350 ± 0.0014 | 4,028M | 7.6 GB | 4 |

± is the standard deviation of the 3 passes (dividing by n): the spread of the stochastic estimate for one checkpoint, not variance across training runs.

### LoRA matches full fine-tuning, on both metrics

Full FT minus LoRA is **+2.4 points of success rate, with a standard error of 3.1 points
(z = 0.76, two-sided p = 0.45)** — not significant. The standard error of the gap is
√(SE_LoRA² + SE_full²) = √(2.2² + 2.2²), i.e. an unpooled two-proportion z-test. That test
treats the two runs as independent. Both actually started from the same 500 initial states;
a paired McNemar test on the per-episode outcomes (87 episodes only LoRA solved, 99 only full
FT solved) gives z = 0.88, so the verdict is the same. The
validation-loss gap is 0.0006, inside the ±0.002 spread between seeded passes. A LoRA
training 0.31% of the parameters reaches the same quality as fine-tuning all 4.03B, with a
160x smaller artefact and a quarter of the hardware.

This compares one training run per arm. Training-seed variance was not measured, so the
result shows these two checkpoints tie. It is suggestive, not proof, that the two methods do.

The two are not, however, the *same* policy. They tie in aggregate while differing sharply
on individual tasks. Per-task two-proportion z-tests (50 vs 50 episodes, z = (full FT −
LoRA) / SE) find two differences that survive Bonferroni correction across the 10 tasks
(|z| > 2.81, i.e. α = 0.05/10 two-sided):
task 5 (this adapter 30% vs full FT 6%, z = -3.29) and task 9 (34% vs 62%, z = +2.92). Their
strengths happen to cancel.

### How to read these numbers against published results

**Do not compare 54.2% directly to the ~96% reported for pi0 on LIBERO-Spatial.** That figure
comes from full fine-tuning with openpi's own recipe and tuning budget. This is a 12.65M
parameter LoRA trained for 67 minutes on 392 episodes, as the cheap arm of a
LoRA-vs-full-fine-tuning comparison, not an attempt to reproduce the paper.

Likewise, the 0.0% base row is the honest **zero-shot** floor: `lerobot/pi0` as released was
never trained on LIBERO and has never seen this embodiment or action space. It is not the
paper's number.

`n_action_steps=5` follows openpi's LIBERO replan interval; it was not tuned for this
checkpoint and is a plausible lever on the result.

## Usage

```python
import torch
from peft import PeftModel
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

meta = LeRobotDatasetMetadata("lerobot/libero_spatial_image")
cfg = PI0Config(pretrained_path="lerobot/pi0", device="cuda", dtype="bfloat16")

policy = make_policy(cfg, ds_meta=meta)
policy = PeftModel.from_pretrained(policy, "dmnsjk/pi0-libero-spatial-lora")
policy.to("cuda").eval()

preprocessor, postprocessor = make_pre_post_processors(
    cfg,
    pretrained_path="lerobot/pi0",
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "stats": meta.stats,
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
        },
    },
)

# One control step. Images are float32 CHW in [0, 1]; state is the 8-D vector above.
batch = {
    "observation.images.image": img,          # (1, 3, 256, 256)
    "observation.images.wrist_image": wrist,  # (1, 3, 256, 256)
    "observation.state": state,               # (1, 8)
    "task": ["pick up the black bowl on the stove and place it on the plate"],
}
with torch.autocast("cuda", dtype=torch.bfloat16):
    action = policy.select_action(preprocessor(batch))
action = postprocessor(action)  # (1, 7), unnormalized
```

Note the normalization statistics: they come from the training dataset's `meta.stats`. A
policy deployed with different statistics than it was trained with will produce
systematically wrong action magnitudes.

## Licence and access

- This adapter and its base model `lerobot/pi0` are **Apache-2.0**.
- pi0 loads its tokenizer/processor from
  [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224), which
  is **gated under the Gemma licence**. You must accept that licence on the Hub and be
  authenticated, or the first call fails with a 403. This is a requirement of the base
  model, not of the adapter.

## Limitations

- **Single suite.** Trained only on LIBERO-Spatial, whose 10 tasks are all "pick up the
  black bowl ... and place it on the plate" with varying spatial referents. Generalization to other LIBERO suites, other embodiments, or the real world is untested here.
- **Simulation only.** All numbers are from the LIBERO MuJoCo simulator.
- **Validation set doubles as the selection set.** The 40 held-out episodes were used both
  to pick the best epoch and to report validation loss, so the loss figure is mildly selection-biased. The success rates are not affected by this (they are measured on the simulator's own initial-state set, not on dataset episodes).
- The final-epoch adapter is worse than this one and is not published. Best-epoch selection is doing real work here.

## Citation

Base model: Black et al., *pi0: A Vision-Language-Action Flow Model for General Robot
Control* (2024). Benchmark: Liu et al., *LIBERO: Benchmarking Knowledge Transfer for
Lifelong Robot Learning* (NeurIPS 2023).
