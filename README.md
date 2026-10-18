# Distributed Fine-Tuning of a Vision-Language-Action Model

End-to-end distributed training pipeline for fine-tuning a Vision-Language-Action (VLA) model
for robotic manipulation, built for a multi-GPU SLURM cluster.

This is a code that was used to finetune the two $\pi_{0}$ weight checkpoints on LIBERO-Spatial available on HF: 

1. LoRA tuning: [pi0-libero-spatial-lora](https://huggingface.co/dmnsjk/pi0-libero-spatial-lora). [Model card](model_cards/pi0-libero-spatial-lora.md).
2. Full Finetuning: [pi0-libero-spatial-full](https://huggingface.co/dmnsjk/pi0-libero-spatial-full). [Model card](model_cards/pi0-libero-spatial-full.md).



`lerobot`'s implementation of π0 is used. π0 is finetuned on **LIBERO** manipulation benchmark.

## Layout

```
train_pi0.py                         # Phase 2 entrypoint: pi0 + LoRA on LIBERO, native DDP
train_pi0_fsdp.py                    # Phase 2B entrypoint: pi0 full FT under FSDP, via accelerate
src/dist_utils.py                    # process group setup, rank gating, cross-rank reductions
src/paths.py                         # $VLA_ROOT / $VLA_RUNS resolution, shared with the shell
scripts/verify_env.py                # checks the pieces that have historically broken here
scripts/eval_adapter.py              # seeded, repeated val loss for an adapter or a full checkpoint
scripts/eval_libero.py               # closed-loop LIBERO rollouts: success rate, 10 tasks x 50 trials
scripts/render_results.py            # success-rate tables for the model cards, from results/*.json
scripts/publish.sh                   # publish both checkpoints to the Hub (dry run unless --go)
scripts/publish_to_hub.py            # upload one checkpoint with its card from model_cards/
slurm/common.sh                      # shared job prologue: paths, caches, .env, venv activation
slurm/pi0_train.sbatch               # Phase 2: 8-GPU production job
slurm/pi0_scaling.sbatch             # Phase 2: 1/2/4-GPU throughput at fixed effective batch
slurm/pi0_fsdp_bench.sbatch          # Phase 2B: sharding memory/throughput sweep
slurm/pi0_full_train.sbatch          # Phase 2B: full-FT convergence run
slurm/pi0_compare_checkpoints.sbatch # base vs LoRA vs full FT val loss, one protocol
slurm/pi0_libero_eval.sbatch         # base vs LoRA vs full FT success rate on LIBERO-spatial
slurm/pi0_libero_reference.sbatch    # lerobot's own LIBERO-finetuned pi0 through our harness
results/                             # per-episode rollout records behind the published success rates
model_cards/                         # the Hub READMEs of both published checkpoints
checkpoints.sha256                   # hashes of the archived checkpoints that were evaluated
IMPROVEMENTS.md                      # the gap to published pi0 LIBERO numbers: causes, costs, order
docs/notes.md                        # technical notes: non-obvious constraints, measured results
pyproject.toml                       # dependencies + CUDA wheel index pins
uv.lock                              # exact pinned versions
constraints.txt                      # pins for installing the LIBERO simulator into the venv
```

The Phase 2 and Phase 2B entrypoints sit side by side: `train_pi0.py` stays the verified
production recipe, and `train_pi0_fsdp.py` builds beside it and imports its dataset split,
loaders and evaluation function, so a full-FT number and a LoRA number are measured on exactly
the same held-out episodes.

## Setup
Requires [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

1.By design, all downloaded content lives under one directory, `$VLA_ROOT` — the virtualenv, the HF and Triton
caches, the FFmpeg prefix and all run outputs. It defaults to `work/` inside the repository. You can adjust this path by exporting VLA_ROOT variable:


```bash
export VLA_ROOT=/path/to/work
export UV_PROJECT_ENVIRONMENT=$VLA_ROOT/venvs/vla
uv sync
```

See [Where things are configured](#where-things-are-configured) below for the full resolution order.


2.**If your system already provides FFmpeg 7 shared libraries, this step is unnecessary.** `lerobot` decodes trajectory video through `torchcodec`, which needs FFmpeg shared
libraries. torchcodec's own
documentation recommends conda-forge; `micromamba` is a fast single-binary client for it:
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
./bin/micromamba -r $VLA_ROOT/.mamba create -y -p $VLA_ROOT/ffmpeg \
  -c conda-forge "ffmpeg=7.*"
```

Jobs put it on the library path (`slurm/common.sh` already does this, and skips it when the
system provides FFmpeg 7):

```bash
export LD_LIBRARY_PATH=$VLA_FFMPEG_PREFIX/lib:${LD_LIBRARY_PATH:-}
```


3.Code reads HF and WANDB credentials from a `.env` file in the repository root. Login to those services separately using CLI or use the `.env`:
```bash
cat > .env <<'EOF'
HF_TOKEN=<hf_token>
WANDB_API_KEY=<wandb_api_key>
EOF
chmod 600 .env
```

You must also accept the licence for [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) with the same HF account. π0's tokenizer is loaded from that gated repo.


After setup, verify if the pieces actually work here:

```bash
source slurm/common.sh        # venv + caches + .env; no manual activation needed
python scripts/verify_env.py
```

### SLURM setup

No job script carries an account or a partition. Set them once per shell and every script submits unchanged:
```bash
export SBATCH_PARTITION=<gpu-partition>
export SBATCH_ACCOUNT=<account>        # only if your default account is not the right one
```



### Simulator Setup -  Evaluation Only

Closed-loop rollouts (`scripts/eval_libero.py`, `slurm/pi0_libero_eval.sbatch`) need LIBERO,
which has to be installed by hand instead of declared in `pyproject.toml`. The isue is with
`robomimic`: it requires `egl-probe`, which ships no wheels and compiles against EGL
development headers. Those come from a `-devel` package, and because `uv sync` resolves and builds the whole graph in one go
it would take the training environment down with it. Installing LIBERO anyway means passing
`--no-deps`.

```bash
source slurm/common.sh                  # REQUIRED: uv pip needs the venv active
uv pip install -c constraints.txt --no-deps hf-libero==0.1.4 robomimic==0.2.0
uv pip install -c constraints.txt \
  "hydra-core>=1.2,<1.4" "robosuite==1.4.0" "bddl==1.0.1" easydict einops thop \
  "matplotlib>=3.5.3" "cloudpickle>=2.0.0" "mujoco>=3.0,<3.9" "future>=0.18.2" \
  h5py psutil tqdm termcolor tensorboard tensorboardX imageio imageio-ffmpeg
uv pip uninstall opencv-python && \
  uv pip install -c constraints.txt --reinstall-package opencv-python-headless \
    "opencv-python-headless==4.12.0.88"
```


**`-c constraints.txt` is not optional.** The constraints pin
torch/torchvision/torchcodec/transformers/numpy/lerobot/accelerate/peft/huggingface-hub to
what is already installed, so the install only adds.


> **`uv sync` will delete all of this.** It removes packages not in `uv.lock`, and none of the sim stack is in there. After installing the simulator, use **`uv sync --inexact`** from then on, or re-run the block above.
#### LIBERO Assets Setup


Download the assets from Hugging Face to chosen directory and link them to LIBERO cache:

```bash
source slurm/common.sh
echo "$VLA_ROOT"
hf download lerobot/libero-assets --repo-type dataset \
  --local-dir "$VLA_ROOT/libero_assets"
mkdir -p ~/.cache/libero
ln -sfn "$VLA_ROOT/libero_assets" ~/.cache/libero/assets
```



Create LIBERO config file:
```bash
source slurm/common.sh
SITE=$(python -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
mkdir -p "$LIBERO_CONFIG_PATH"
cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
assets: $VLA_ROOT/libero_assets
bddl_files: $SITE/libero/libero/bddl_files
benchmark_root: $SITE/libero/libero
datasets: $SITE/libero/libero/../datasets
init_states: $SITE/libero/libero/init_files
EOF
```




### Where Things Are Configured

Every job script starts with one line — `source "${SLURM_SUBMIT_DIR:-.}/slurm/common.sh"` —
and that file resolves the whole environment: it sources `.env`, exports the cache
locations, puts FFmpeg on the library path, and activates the virtualenv.

Each variable is only *defaulted*, so an exported value always wins.
| Variable | Default | What it controls |
| --- | --- | --- |
| `VLA_ROOT` | `work/` in the repository | work root for everything below |
| `VLA_VENV` | first of `$UV_PROJECT_ENVIRONMENT`, `$VIRTUAL_ENV`, `$VLA_ROOT/venvs/vla`, `./.venv` that exists | the virtualenv jobs activate |
| `VLA_RUNS` | `$VLA_ROOT/runs` | where training **writes** checkpoints (the entrypoints' default `--out`) |
| `VLA_FFMPEG_PREFIX` | `$VLA_ROOT/ffmpeg` | prefix holding the FFmpeg 7 shared libraries |
| `HF_HOME`, `TRITON_CACHE_DIR`, `WANDB_DIR`, `LIBERO_CONFIG_PATH` | under `$VLA_ROOT` | caches |
| `CKPT_ROOT` | an archive on permanent storage, else `$VLA_RUNS` | where evaluation reads finished checkpoints from |
| `NCCL_SOCKET_IFNAME` | `ib0` | network interface NCCL uses to set up multi-GPU communication |




## Running

`sbatch` needs nothing prepared — the job script sources `slurm/common.sh` itself. The
plain `python ...` commands below assume you have run `source slurm/common.sh` first, which
activates the venv and exports `$VLA_RUNS`.

### π0 + LoRA on LIBERO

```bash
sbatch slurm/pi0_train.sbatch    # 8 GPUs, 7 epochs
```

Evaluate a saved adapter. **Always pass `--repeats`**: π0 is a flow-matching model, so
validation loss is a stochastic estimate and a single number is misleading.

```bash
python scripts/eval_adapter.py --adapter $VLA_RUNS/pi0_libero/adapter_best \
  --repeats 5
```


### π0 full fine-tuning, sharded with FSDP

```bash
sbatch slurm/pi0_full_train.sbatch      # 8 GPUs, full FT to convergence
```

The exported checkpoint
is a complete π0 policy directory, so it evaluates through the same script as the adapter:

```bash
python scripts/eval_adapter.py --policy $VLA_RUNS/pi0_full/model_best \
  --repeats 5
```


Iterating locally on one GPU, without SLURM:

```bash
source slurm/common.sh      # venv + caches + .env, same as a job
python train_pi0.py --max-steps 20 --val-episodes 2 --grad-checkpointing \
  --out $VLA_RUNS/pi0_dev
```




### Computational requirements comparison
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
  root FSDP unit — a consequence of the wrap policy below — and stays gathered
  across forward and backward, so ZeRO-3 cannot shard the part that dominates. (This was
  checked against a measurement artefact: the fp32 model is built on the GPU before sharding,
  costing 15.5 GiB, but that is below the reported peaks, so they are genuine training peaks.)
- **Gradient checkpointing is 60% *faster* here, not slower** (49.7 vs 31.0 samples/s) — the
  opposite sign to the LoRA run's ~13% cost. At 33.2 GiB on a 40 GiB card the allocator is
  under enough pressure that recomputation wins back more than it spends.
- **LoRA under FSDP peaks at 11.7 GiB, exactly matching the LoRA+DDP production run** — two
  independent code paths agreeing on the same number.


One finding worth stating plainly: **the textbook FSDP wrap policy does not apply to π0.**
FSDP gathers a unit's parameters in a pre-forward hook on the wrapped module, but π0 never
calls its decoder layers' `forward`. It runs its own fused loop (`compute_layer_complete` in
lerobot's `modeling_pi0.py`) so the VLM prefix and the action expert can share one attention
op, and calls `layer.input_layernorm`, `layer.self_attn.q_proj/k_proj/v_proj/o_proj` and
`layer.mlp` directly. A unit wrapped around `GemmaDecoderLayer` would never have its hook
fire, so its parameters would still be sharded when the loop reads them. We expect this to
fail loudly (with `use_orig_params=True` the sharded weights have the wrong shape for
`F.linear`), but we never ran it: the policy was ruled out by reading the code, not by a
failed job. The policy we use wraps `GemmaMLP` and `SiglipEncoderLayer`, which *are* invoked
as modules. That covers 2.45B of 4.04B parameters, and the rest falls into the root unit,
which is still sharded. This is the coarsest correct choice, not the only one: the
attention projections and layernorms are also called as modules and could be wrapped too,
which would shrink the root unit. We didn't try that, so the ZeRO-2/ZeRO-3 result above
reflects this policy and isn't a hard limit of the model. For the same reason we expect
DeepSpeed ZeRO-3, which hooks every submodule down to each `nn.Linear`, to work on π0. That
is untested.


