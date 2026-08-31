"""Publish a checkpoint to the Hugging Face Hub with the repo's own model card.

The card lives in `model_cards/` rather than inside the run directory because PEFT
regenerates `adapter_best/README.md` on every export, so anything hand-written there is one
re-export away from being replaced by a stub full of `[More Information Needed]`.

Nothing is staged to a temp directory. The checkpoint is uploaded in place with its own
README ignored, and the card is uploaded separately as `README.md`. That avoids copying a
7.6 GB checkpoint through `$TMPDIR` and, just as importantly, works when the checkpoint tree
is read-only -- which it should be, since these are the artefacts of runs nobody wants to
repeat.

    python scripts/publish_to_hub.py --dry-run \
        --checkpoint $PLG_GROUPS_STORAGE/plgguseby/dmns/vla/pi0_libero/adapter_best \
        --card model_cards/pi0-libero-spatial-lora.md \
        --repo-id dmnsjk/pi0-libero-spatial-lora --private
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    # NOTE: private only takes effect when the repo is CREATED. exist_ok=True will not
    # downgrade an already-public repo; use HfApi.update_repo_settings for that.
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="list what would upload, upload nothing")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        sys.exit("HF_TOKEN is not set (it lives in .env, which is gitignored)")
    if not args.checkpoint.is_dir():
        sys.exit(f"no such checkpoint directory: {args.checkpoint}")
    if not args.card.is_file():
        sys.exit(f"no such card: {args.card}")

    files = sorted(f for f in args.checkpoint.rglob("*") if f.is_file())
    payload = [f for f in files if f.name != "README.md"]
    total = sum(f.stat().st_size for f in payload)

    print(f"{args.repo_id}  (private={args.private})")
    print(f"  card: {args.card}  ({args.card.stat().st_size / 1024:.1f} KiB) -> README.md")
    for f in payload:
        print(f"  {f.stat().st_size / 2**20:9.2f} MiB  {f.relative_to(args.checkpoint)}")
    skipped = [f for f in files if f.name == "README.md"]
    for f in skipped:
        print(f"  {'(skipped)':>9}    {f.relative_to(args.checkpoint)}  <- replaced by the card")
    print(f"  total: {total / 2**20:.1f} MiB in {len(payload)} file(s)")

    if args.dry_run:
        print("\n--dry-run: nothing uploaded")
        return

    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=str(args.checkpoint),
        repo_id=args.repo_id,
        repo_type="model",
        ignore_patterns=["README.md"],
    )
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"\nhttps://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
