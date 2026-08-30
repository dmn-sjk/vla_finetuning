"""Publish a checkpoint to the Hugging Face Hub with the repo's own model card.

The card is deliberately kept in the repo (`model_cards/`) rather than edited inside the
run directory: PEFT regenerates `adapter_best/README.md` on every export, so anything
hand-written there is one re-export away from being silently replaced by a stub full of
`[More Information Needed]`.

    python scripts/publish_to_hub.py \
        --checkpoint $PLG_GROUPS_STORAGE/plgguseby/dmns/vla/pi0_libero/adapter_best \
        --card model_cards/pi0-libero-spatial-lora.md \
        --repo-id dmnsjk/pi0-libero-spatial-lora --private
"""

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--card", type=Path, required=True)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="stage and list, upload nothing")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        sys.exit("HF_TOKEN is not set (it lives in .env, which is gitignored)")
    if not args.checkpoint.is_dir():
        sys.exit(f"no such checkpoint directory: {args.checkpoint}")
    if not args.card.is_file():
        sys.exit(f"no such card: {args.card}")

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "repo"
        shutil.copytree(args.checkpoint, stage)
        # Overwrite whatever card the exporter left behind.
        shutil.copyfile(args.card, stage / "README.md")

        files = sorted(f.relative_to(stage) for f in stage.rglob("*") if f.is_file())
        total = sum((stage / f).stat().st_size for f in files)
        print(f"staged {len(files)} files, {total / 2**20:.1f} MiB, for {args.repo_id}:")
        for f in files:
            print(f"  {(stage / f).stat().st_size / 2**20:8.2f} MiB  {f}")
        if args.dry_run:
            print("\n--dry-run: nothing uploaded")
            return

        api = HfApi(token=token)
        api.create_repo(args.repo_id, private=args.private, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(stage), repo_id=args.repo_id, repo_type="model")
        print(f"\nhttps://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
