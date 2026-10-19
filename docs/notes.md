# Technical notes

Non-obvious facts about pi0 and LIBERO.

## pi0 under FSDP

- pi0 never calls `GemmaDecoderLayer.forward`: `compute_layer_complete` in lerobot's `modeling_pi0.py` reaches into `layer.self_attn.q_proj(...)` directly so the VLM prefix and the action expert share one attention op. FSDP unshards in a module's pre-forward hook, so wrapping decoder layers would leave parameters sharded while the math runs.
- The wrap policy targets `GemmaMLP` and `SiglipEncoderLayer` instead (both are called as modules). Coverage is ~61% of parameters (`fsdp_wrapped=` on startup); the rest sits in the root unit. `train_pi0_fsdp.py` raises below 50% so a transformers class rename cannot collapse FULL_SHARD into SHARD_GRAD_OP unnoticed.
- DeepSpeed ZeRO-3 hooks every submodule down to each `nn.Linear`, so it is expected to work on pi0, but is untested. FSDP SHARD_GRAD_OP is ZeRO-2 without the extra dependency.
- accelerate ignores the FSDP plugin at world size 1 (the branch sits inside the MULTI_GPU arm), so a single-process `--sharding full_shard` would train unsharded. The script refuses it.
- The model is built in fp32: FSDP1 flattens each unit into one tensor and cannot mix dtypes, and pi0's own bf16 mode keeps the vision tower and norms in fp32. bf16 compute comes from an FSDP `MixedPrecision` policy (fp32 master weights and gradient reduction).
- No `no_sync()` on accumulation micro-batches under FSDP: it keeps unsharded gradients resident for the whole window.
- The LR scheduler is not passed through `accelerate.prepare`: `AcceleratedScheduler` steps once per process, multiplying the schedule by world size. It is stepped by hand.
- `save_state` serialises the full 4.03B model even for LoRA (~16 GB per epoch). It is a resume point, not an artefact; `--no-checkpoint` skips it.
- The exported directory must include both processor configs: they carry the normalization statistics, without which the weights load and predict nonsense.

## pi0 + LoRA under DDP

- LoRA on the VLM tower needs `find_unused_parameters=True`. pi0 consumes only the prefix KV cache, so the final language-model `o_proj` feeds nothing and its `lora_B` pair (2 of 298 tensors) gets no gradient. Changing `--lora-target` changes what reaches the DDP reducer, so re-run the distributed checks after changing it.
- PEFT fullmatches string `target_modules` as a regex; a typo matches nothing. `train_pi0.py` raises if fewer than 1M parameters are trainable.
- The vision tower is excluded from every LoRA preset.
- Gradient checkpointing under DDP needs `use_reentrant=False`.
- Pass `device_id=` to `init_process_group`, or `barrier()` can pick the wrong GPU.

## Data and validation

- `lerobot/libero_spatial_image`: 432 episodes, 52,970 frames, 10 tasks, stored as PNG (not video). Dataloading is CPU-bound PNG decode on Lustre; give it workers.
- Split validation by episode, not frame: adjacent frames are near-duplicates.
- Val loss is stochastic (flow matching samples a timestep and noise per forward). Use `python -m scripts.eval_adapter --repeats N` and report mean +- std. Differences below ~0.005 are noise. `evaluate()` offsets the seed per rank, otherwise ranks draw identical noise.
- Val loss only exercises `policy.forward()`. It says nothing about the inference path (unnormalization, KV-cache prefill, Euler sampling in `sample_actions`).

## LIBERO evaluation

- The dataset port rotates robosuite frames 180 degrees. Unrotated frames run without error and score near zero. `eval_libero.py --verify-obs` scores each transform against a real frame; rot180 wins by an order of magnitude on both cameras.
- The observation format is the dataset's (`observation.images.image` / `.wrist_image`, one 8-D state), not lerobot's `LiberoEnv` (`.image2`, dict state). The script drives `OffScreenRenderEnv` directly.
- The post-processor must receive the training dataset's stats too, or actions reach the simulator as z-scores. `_assert_unnormalizes()` checks that a zero action maps back to the dataset mean.
- Normalization stats always come from the training dataset, even for other suites.
- `meta/info.json` labels the state as position + quaternion + gripper. The real layout is `[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]`; trust the stats, not the names.
- Recorded actions are normalized OSC_POSE deltas; gripper is -1/1. The action-to-metre slope is ~0.01, not the controller's `output_max` of 0.05.
- LoRA adapters are evaluated unmerged: merging in bf16 rounds the weights.

## Sim-stack install

- `hf-libero` pins `robomimic==0.2.0`, which depends on `egl-probe`; it compiles against EGL headers the login node lacks. `hf-egl-probe` fails the same way.
- `egl_probe` is only imported by robomimic env wrappers LIBERO does not use, hence `--no-deps` on `hf-libero`/`robomimic` with their real dependencies added by hand. This is why `lerobot[libero]` is not in `pyproject.toml`.
- `constraints.txt` keeps the install additive; it is a `uv pip` flag only (`uv sync` ignores it).
- `uv sync` is exact and deletes the sim stack; use `--inexact` afterwards.
- robosuite pulls `opencv-python`, which shares the `cv2` module with `opencv-python-headless`; it must be uninstalled explicitly. `verify_env.py` checks which distribution owns `cv2`.

## Results (LIBERO-spatial)

LoRA: `--lora-target attention`, r=32, 12.65M trainable. Full FT: 4.028B trainable, FULL_SHARD.

| checkpoint | val loss (n=3) | success (500 eps) | artefact | min GPUs |
| --- | --- | --- | --- | --- |
| base `lerobot/pi0` | 0.8719 +- 0.0048 | 0.0% | - | - |
| LoRA `adapter_best` | 0.3344 +- 0.0020 | 54.2% +- 2.2 | 49 MB | 1 |
| full FT `model_best` | 0.3350 +- 0.0014 | 56.6% +- 2.2 | 7.6 GB | 4 |

The fine-tunes tie (success z = 0.76) but differ per task: task 5 LoRA 30% vs full 6%, task 9 LoRA 34% vs full 62%. Full FT peaks at epoch 0 and overfits back to base-level val loss by epoch 4.

LoRA + DDP scaling, effective batch 128, gradient checkpointing, 11.3 GiB peak at every size:

| GPUs | samples/s | speedup | efficiency |
| --- | --- | --- | --- |
| 1 | 12.9 | 1.00x | 100% |
| 2 | 24.8 | 1.92x | 96% |
| 4 | 46.8 | 3.63x | 91% |

Full FT memory, per-device batch 8, gradient checkpointing, `--reduce-dtype fp32`:

| GPUs | strategy | samples/s | peak VRAM |
| --- | --- | --- | --- |
| 1 | unsharded | OOM | - |
| 2 | FULL_SHARD | OOM | - |
| 8 | unsharded (DDP) | OOM | - |
| 4 | SHARD_GRAD_OP | 18.6 | 25.1 GiB |
| 4 | FULL_SHARD | 19.3 | 25.1 GiB |
| 8 | SHARD_GRAD_OP | 48.4 | 17.0 GiB |
| 8 | FULL_SHARD | 49.7 | 17.0 GiB |
| 8 | FULL_SHARD, no grad ckpt | 31.0 | 33.2 GiB |
| 8 | FULL_SHARD, LoRA | 50.9 | 11.7 GiB |

ZeRO-2 peak at world size N: 16.1/N fp32 master + 8.1 bf16 params + 16.1/N grads + 32.2/N AdamW + ~1 activations, matching both rows. ZeRO-3 gives no extra saving because 39% of the model sits in the always-gathered root unit.
