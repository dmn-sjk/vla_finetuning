"""Flags shared by train_pi0.py and train_pi0_fsdp.py."""

from src.pi0 import LORA_TARGETS


def add_common_args(p) -> None:
    p.add_argument("--dataset", default="lerobot/libero_spatial_image")
    p.add_argument("--policy", default="lerobot/pi0", help="base weights")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=0, help="0 = no cap; for smoke tests")
    p.add_argument("--batch-size", type=int, default=8, help="per device")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-episodes", type=int, default=40, help="held out by episode, not by frame")
    p.add_argument(
        "--lora-target",
        choices=sorted(LORA_TARGETS),
        default="attention",
        help="which modules get adapters; see LORA_TARGETS",
    )
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--no-bf16", dest="bf16", action="store_false", default=True)
    p.add_argument("--resume", action="store_true", help="resume from the checkpoint in --out")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="optimizer steps before throughput timing starts",
    )
