#!/usr/bin/env bash
# Publish the LIBERO checkpoints to the Hugging Face Hub. Dry run unless --go.
set -euo pipefail

usage() { echo "usage: scripts/publish.sh [--go] [--public] [lora|full|all]..."; }

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) && cd "$REPO_ROOT"

GO=0; PRIVATE="--private"; TARGETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --go)       GO=1 ;;
    --public)   PRIVATE="" ;;
    -h|--help)  usage; exit 0 ;;
    lora|full)  TARGETS+=("$1") ;;
    all)        TARGETS+=(lora full) ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(lora full)

source "$REPO_ROOT/slurm/common.sh"
CK="${CKPT_ROOT:-$VLA_RUNS}"
HF_USER="${HF_USER:-dmnsjk}"
declare -A CKPT=( [lora]="$CK/pi0_libero/adapter_best" [full]="$CK/pi0_full/model_best" )
declare -A NAME=( [lora]="pi0-libero-spatial-lora" [full]="pi0-libero-spatial-full" )

[[ $GO -eq 1 && -z "${HF_TOKEN:-}" ]] && { echo "HF_TOKEN is not set (see .env)" >&2; exit 1; }

for t in "${TARGETS[@]}"; do
  cmd=(python -m scripts.publish_to_hub --checkpoint "${CKPT[$t]}"
       --card "model_cards/${NAME[$t]}.md" --repo-id "$HF_USER/${NAME[$t]}")
  [[ -n "$PRIVATE" ]] && cmd+=("$PRIVATE")
  [[ $GO -eq 0 ]] && cmd+=(--dry-run)
  "${cmd[@]}"
done
