"""pi0 helpers shared by the training and evaluation entrypoints."""

import contextlib
import os
from pathlib import Path

import torch
from lerobot.policies.factory import make_pre_post_processors

from src import dist_utils as D

# PEFT fullmatches a string target_modules against the module path, hence the `.*` anchors.
# The vision tower is excluded on purpose: SigLIP features transfer well and are easy to damage.
_PROJECTIONS = "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
_TOWERS = "gemma_expert|language_model"

LORA_TARGETS = {
    # lerobot default: action-expert q/v
    "default": rf"(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.({_PROJECTIONS}))",
    # q,k,v,o on both towers
    "attention": rf"(.*\.({_TOWERS})\..*\.self_attn\.(q|k|v|o)_proj|model\.({_PROJECTIONS}))",
    # attention + MLP on both towers
    "attention-mlp": (
        rf"(.*\.({_TOWERS})\..*\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
        rf"|model\.({_PROJECTIONS}))"
    ),
}


def wrap_lora(policy, target: str, r: int, alpha: int, dropout: float):
    return policy.wrap_with_peft(
        peft_cli_overrides={
            "target_modules": LORA_TARGETS[target],
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": dropout,
        }
    )


def make_processors(cfg, meta, device, pretrained_path, unnormalize: bool = False):
    """Pre/post-processors normalizing with `meta`'s stats; the unnormalizer too if asked."""
    norm = {
        "stats": meta.stats,
        "features": {**cfg.input_features, **cfg.output_features},
        "norm_map": cfg.normalization_mapping,
    }
    extra = {"postprocessor_overrides": {"unnormalizer_processor": norm}} if unnormalize else {}
    return make_pre_post_processors(
        cfg,
        pretrained_path=pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": device.type},
            "normalizer_processor": norm,
        },
        **extra,
    )


def autocast_ctx(enabled: bool):
    """Zero-arg factory for a bf16 autocast context, or a no-op one."""
    if enabled:
        return lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext


def atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(policy, loader, preprocessor, autocast, device, seed: int = 1234):
    """Mean val loss across ranks.

    Stochastic (flow matching), so seeded per rank; RNG state is restored afterwards.
    """
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    torch.manual_seed(seed + 10_000 * D.rank())

    policy.eval()
    loss_sum, n_seen = 0.0, 0
    for raw in loader:
        batch = preprocessor(raw)
        with autocast():
            loss, _ = policy(batch)
        bs = batch["action"].shape[0]
        loss_sum += loss.item() * bs
        n_seen += bs
    # reduce sums, not means: ranks may hold different sample counts
    total_loss = D.all_reduce_sum(loss_sum, device)
    total_n = D.all_reduce_sum(float(n_seen), device)
    policy.train()

    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)
    return total_loss / max(total_n, 1.0)
