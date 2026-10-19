---
base_model: lerobot/pi0
library_name: lerobot
license: apache-2.0
pipeline_tag: robotics
tags:
  - lerobot
  - robotics
  - vision-language-action
  - pi0
  - libero
  - fsdp
datasets:
  - lerobot/libero_spatial_image
---

# pi0-libero-spatial-full

**All 3.3B parameters** of [`lerobot/pi0`](https://huggingface.co/lerobot/pi0) fine-tuned on the **LIBERO-Spatial** suite, sharded with PyTorch FSDP (ZeRO-3).

> This checkpoint is published as the control arm of a full-fine-tuning vs. LoRA comparison, not because it is the better policy. It is tied with a [49 MB LoRA adapter](https://huggingface.co/dmnsjk/pi0-libero-spatial-lora) that trains on a quarter of the hardware, and it is 160x larger. If you just want a LIBERO-Spatial pi0, use the adapter. What this checkpoint is useful for is studying what full fine-tuning does to a 3.3B VLA on a small suite: what it does is overfit, within a single epoch.

## Model details

| | |
| --- | --- |
| Base model | `lerobot/pi0` (PaliGemma-3B backbone + flow-matching action expert) |
| Method | Full fine-tuning, all 3.3B parameters trainable (lerobot's port counts 4,028M: it also holds ~0.79B of vocabulary-sized LM heads that π0 never uses) |
| Dataset | [`lerobot/libero_spatial_image`](https://huggingface.co/datasets/lerobot/libero_spatial_image), 432 episodes / 52,970 frames, 10 tasks |
| Split | 392 train / 40 validation, held out **by episode** |
| Artefact | 7.6 GB (`model.safetensors` + processor configs) |
| Observations / actions | two 256x256 cameras, 8-D state, 7-D delta action, 50-step chunk at 10 Hz |

## Training

| | |
| --- | --- |
| Code | [dmn-sjk/vla_finetuning](https://github.com/dmn-sjk/vla_finetuning): `train_pi0_fsdp.py`, launched by `slurm/pi0_full_train.sbatch` |
| Hardware | 4x NVIDIA A100-40GB, FSDP `FULL_SHARD` (ZeRO-3) |
| Wall clock | 3 h 20 min, 19.2 samples/s, 25.1 GiB peak per GPU |
| Epochs | 5 (this checkpoint is **epoch 0** — see below) |
| Effective batch | 256 (8 per device x 8 accumulation x 4 ranks) |
| Optimizer | AdamW, lr 2.5e-5, 50-step warmup, cosine schedule |
| Precision | bf16 compute/all-gather, fp32 master weights and gradient reduction |
| Other | gradient checkpointing on |


## Results

Two independent measurements, both on held-out data.

### Closed-loop success rate (the metric the literature reports)

500 episodes: all 10 LIBERO-spatial tasks x 50 trials, each trial starting from the suite's own *i*-th saved initial state, 280-step cap, `n_action_steps=5`. This is the standard LIBERO/OpenVLA evaluation protocol.

| checkpoint | success rate ± SE | successes |
| --- | --- | --- |
| base `lerobot/pi0` (zero-shot) | 0.0% ± 0.0 | 0/500 |
| [LoRA `adapter_best`](https://huggingface.co/dmnsjk/pi0-libero-spatial-lora) | **54.2% ± 2.2** | 271/500 |
| **full fine-tune `model_best` (this model)** | **56.6% ± 2.2** | 283/500 |

± is the **binomial standard error** of each rate, √(p(1−p)/n) with n = 500. It measures evaluation noise for this one checkpoint (how much the rate would move on a re-run), not variance across training seeds. For the 0/500 zero-shot row it is 0 by construction; a 95% Wilson interval puts the true rate between 0% and 0.8%.

Per task:

| task | instruction | LoRA adapter | this model |
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

| checkpoint | val loss (mean ± std, n=3) | trainable params | artefact | min GPUs |
| --- | --- | --- | --- | --- |
| base `lerobot/pi0` (zero-shot) | 0.8719 ± 0.0048 | — | — | — |
| [LoRA `adapter_best`](https://huggingface.co/dmnsjk/pi0-libero-spatial-lora) | 0.3344 ± 0.0020 | 12.65M | 49 MB | 1 |
| **full fine-tune `model_best` (this model)** | **0.3350 ± 0.0014** | 3.3B | 7.6 GB | 4 |

± is the standard deviation of the 3 passes (dividing by n): the spread of the stochastic estimate for one checkpoint, not variance across training runs.

### LoRA matches full fine-tuning, on both metrics

Full FT minus LoRA is **+2.4 points of success rate, with a standard error of 3.1 points (z = 0.76, two-sided p = 0.45)** — not significant. The standard error of the gap is √(SE_LoRA² + SE_full²) = √(2.2² + 2.2²), i.e. an unpooled two-proportion z-test. That test treats the two runs as independent. Both actually started from the same 500 initial states; a paired McNemar test on the per-episode outcomes (87 episodes only LoRA solved, 99 only full FT solved) gives z = 0.88, so the verdict is the same. The validation-loss gap is 0.0006, inside the ±0.002 spread between seeded passes. A LoRA training 0.38% of the parameters reaches the same quality as fine-tuning all 3.3B, with a 160x smaller artefact and a quarter of the hardware.

This compares one training run per arm. Training-seed variance was not measured, so the result shows these two checkpoints tie. It is suggestive, not proof, that the two methods do.

The two are not, however, the *same* policy. They tie in aggregate while differing sharply on individual tasks. Per-task two-proportion z-tests (50 vs 50 episodes, z = (full FT − LoRA) / SE) find two differences that survive Bonferroni correction across the 10 tasks (|z| > 2.81, i.e. α = 0.05/10 two-sided): task 5 (this adapter 30% vs full FT 6%, z = -3.29) and task 9 (34% vs 62%, z = +2.92). Their strengths happen to cancel.

### How to read these numbers against published results

**Do not compare 54.2% directly to the ~96% reported for pi0 on LIBERO-Spatial.** That figure comes from full fine-tuning with openpi's own recipe and tuning budget. This is a 12.65M parameter LoRA trained for 67 minutes on 392 episodes, as the cheap arm of a LoRA-vs-full-fine-tuning comparison, not an attempt to reproduce the paper.

Likewise, the 0.0% base row is the honest **zero-shot** floor: `lerobot/pi0` as released was never trained on LIBERO and has never seen this embodiment or action space. It is not the paper's number.

`n_action_steps=5` follows openpi's LIBERO replan interval; it was not tuned for this checkpoint and is a plausible lever on the result.

## The overfitting curve

Validation loss over the 5 epochs, against a falling training loss:

| epoch | val loss | train loss |
| --- | --- | --- |
| 0 | **0.3344** (best) | ~0.35 |
| 1 | 0.3658 | |
| 2 | 0.5015 | |
| 3 | 0.6869 | |
| 4 | 0.8361 | 0.084 |

By epoch 4 the model fits the training set almost perfectly (train loss 0.084) and its validation loss has returned to roughly that of the untrained base model (0.8719 +- 0.0048) — the entire fine-tuning gain given back. 392 episodes is not enough data to move 3.3B parameters for more than about one epoch. Only the epoch-0 checkpoint is published.

## Usage

```python
import torch
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

REPO = "dmnsjk/pi0-libero-spatial-full"
meta = LeRobotDatasetMetadata("lerobot/libero_spatial_image")
cfg = PI0Config(pretrained_path=REPO, device="cuda", dtype="bfloat16")
policy = make_policy(cfg, ds_meta=meta).to("cuda").eval()

preprocessor, postprocessor = make_pre_post_processors(
    cfg,
    pretrained_path=REPO,
    preprocessor_overrides={
        "device_processor": {"device": "cuda"},
        "normalizer_processor": {
            "stats": meta.stats,
            "features": {**cfg.input_features, **cfg.output_features},
            "norm_map": cfg.normalization_mapping,
        },
    },
)

batch = {
    "observation.images.image": img,          # (1, 3, 256, 256) float32 in [0, 1]
    "observation.images.wrist_image": wrist,  # (1, 3, 256, 256)
    "observation.state": state,               # (1, 8)
    "task": ["pick up the black bowl on the stove and place it on the plate"],
}
with torch.autocast("cuda", dtype=torch.bfloat16):
    action = policy.select_action(preprocessor(batch))
action = postprocessor(action)  # (1, 7)
```

## Licence and access

- This checkpoint and its base model `lerobot/pi0` are **Apache-2.0**.
- The tokenizer/processor comes from [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224), which is **gated under the Gemma licence**. Accept it on the Hub and authenticate, or loading fails with a 403.

## Limitations

- **Single suite.** Trained only on LIBERO-Spatial, whose 10 tasks are all "pick up the black bowl ... and place it on the plate" with varying spatial referents. Generalization to other LIBERO suites, other embodiments, or the real world is untested here.
- **Simulation only.** All numbers are from the LIBERO MuJoCo simulator.
- **Validation set doubles as the selection set.** The 40 held-out episodes were used both to pick the best epoch and to report validation loss, so the loss figure is mildly selection-biased. The success rates are not affected by this (they are measured on the simulator's own initial-state set, not on dataset episodes).

## Citation

Base model: Black et al., *pi0: A Vision-Language-Action Flow Model for General Robot Control* (2024). Benchmark: Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning* (NeurIPS 2023).
