"""VLA-shaped synthetic data and a small model.

Phase 1 is about getting the distributed plumbing right, so this stands in for
lerobot datasets and the real VLA backbone. Shapes match what Phase 2 will feed
in — multi-camera frames, proprioceptive state, a tokenized instruction, and a
chunk of continuous actions — so train.py does not change when these are
swapped for the real thing.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset

N_CAMERAS = 2
IMAGE_SIZE = 96
STATE_DIM = 14
ACTION_DIM = 14
ACTION_CHUNK = 8
VOCAB_SIZE = 512
MAX_TOKENS = 16


class ToyTrajectoryDataset(Dataset):
    """Deterministic per-index samples, so every rank agrees on what item i is."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        g = torch.Generator().manual_seed(idx)
        images = torch.randn(N_CAMERAS, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
        state = torch.randn(STATE_DIM, generator=g)
        tokens = torch.randint(0, VOCAB_SIZE, (MAX_TOKENS,), generator=g)
        # Learnable target: a fixed function of the inputs plus noise.
        base = state.sum() + images.mean() * 10 + tokens.float().mean() * 0.01
        actions = base + torch.randn(ACTION_CHUNK, ACTION_DIM, generator=g) * 0.1
        return {"images": images, "state": state, "tokens": tokens, "actions": actions}


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        return x + self.net(x)


class ToyVLA(nn.Module):
    def __init__(self, dim: int = 256, depth: int = 6, grad_checkpointing: bool = False):
        super().__init__()
        self.grad_checkpointing = grad_checkpointing
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=4), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.vision_proj = nn.Linear(64 * N_CAMERAS, dim)
        self.state_proj = nn.Linear(STATE_DIM, dim)
        self.token_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, ACTION_CHUNK * ACTION_DIM)

    def forward(self, images, state, tokens):
        b, n = images.shape[:2]
        vis = self.vision(images.flatten(0, 1)).reshape(b, -1)
        x = self.vision_proj(vis) + self.state_proj(state) + self.token_emb(tokens).mean(1)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                # use_reentrant=False is required for DDP + checkpointing to
                # cooperate; the reentrant path misses gradient hooks.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.head(x).reshape(b, ACTION_CHUNK, ACTION_DIM)
