"""Episode-level train/val split of a LeRobot dataset and its loaders."""

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, DistributedSampler

from src import dist_utils as D


def build_datasets(args, meta, cfg):
    """Split by episode: adjacent frames are near-duplicates."""
    # pi0 predicts action chunks
    delta_timestamps = resolve_delta_timestamps(cfg, meta)

    n_val = min(args.val_episodes, max(1, meta.total_episodes // 10))
    all_eps = list(range(meta.total_episodes))
    val_eps = all_eps[-n_val:]
    train_eps = all_eps[:-n_val]

    train_ds = LeRobotDataset(args.dataset, episodes=train_eps, delta_timestamps=delta_timestamps)
    val_ds = LeRobotDataset(args.dataset, episodes=val_eps, delta_timestamps=delta_timestamps)
    return train_ds, val_ds, len(train_eps), len(val_eps)


def build_loaders(args, train_ds, val_ds):
    train_sampler = val_sampler = None
    if D.is_distributed():
        # drop_last keeps ranks in lockstep
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)

    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, **common)
    return train_loader, val_loader, train_sampler
