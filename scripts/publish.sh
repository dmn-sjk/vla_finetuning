#!/usr/bin/env bash
# Publish the LIBERO checkpoints to the Hugging Face Hub.
#
# DRY RUN BY DEFAULT. Uploading is outward-facing and hard to undo -- a repo that has been
# public for a minute may already be mirrored -- so the upload path needs an explicit --go.
#
#   scripts/publish.sh                 # show exactly what would upload, upload nothing
#   scripts/publish.sh --go            # publish both, private
#   scripts/publish.sh --go lora       # publish only the 49 MB adapter
#   scripts/publish.sh --go --detach   # same, but survives losing the login shell
#
# The 7.6 GB full checkpoint takes a while and login nodes kill long foreground processes;
# use --detach (or tmux) for it. Repos are created private unless --public is passed, and
# note that visibility is only set at CREATION -- see the warning printed below.
set -euo pipefail

# Print the header comment block, however long it grows.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

GO=0; DETACH=0; PRIVATE="--private"; TARGETS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --go)       GO=1 ;;
    --public)   PRIVATE="" ;;
    --detach)   DETACH=1 ;;
    -h|--help)  usage; exit 0 ;;
    lora|full)  TARGETS+=("$1") ;;
    all)        TARGETS+=(lora full) ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=(lora full)

# Re-exec detached, before doing any work, so the caller gets their prompt straight back.
if [[ $DETACH -eq 1 ]]; then
  mkdir -p logs
  LOG="logs/publish-$(date +%Y%m%d-%H%M%S).log"
  args=(); [[ $GO -eq 1 ]] && args+=(--go); [[ -z "$PRIVATE" ]] && args+=(--public)
  nohup "${BASH_SOURCE[0]}" "${args[@]}" "${TARGETS[@]}" > "$LOG" 2>&1 &
  echo "detached as pid $! -> $LOG"
  echo "follow with: tail -f $LOG"
  exit 0
fi

# Sources .env, resolves VLA_ROOT/VLA_RUNS and activates the venv. See slurm/common.sh.
source "$REPO_ROOT/slurm/common.sh"

CK="${CKPT_ROOT:-${PLG_GROUPS_STORAGE:-}/plgguseby/dmns/vla}"
[[ -d "$CK" ]] || CK="$VLA_RUNS"   # not everyone has that group storage
HF_USER="${HF_USER:-dmnsjk}"

declare -A CKPT=( [lora]="$CK/pi0_libero/adapter_best" [full]="$CK/pi0_full/model_best" )
declare -A CARD=( [lora]="model_cards/pi0-libero-spatial-lora.md"
                  [full]="model_cards/pi0-libero-spatial-full.md" )
declare -A NAME=( [lora]="pi0-libero-spatial-lora" [full]="pi0-libero-spatial-full" )

if [[ $GO -eq 1 && -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; it lives in .env, which is gitignored" >&2; exit 1
fi

echo "checkpoint root : $CK"
echo "hub account     : $HF_USER"
echo "visibility      : ${PRIVATE:---public}"
echo "mode            : $([[ $GO -eq 1 ]] && echo 'UPLOAD' || echo 'dry run (pass --go to upload)')"
echo

for t in "${TARGETS[@]}"; do
  echo "======== $t ========"
  cmd=(python scripts/publish_to_hub.py
       --checkpoint "${CKPT[$t]}"
       --card       "${CARD[$t]}"
       --repo-id    "$HF_USER/${NAME[$t]}")
  [[ -n "$PRIVATE" ]] && cmd+=("$PRIVATE")
  [[ $GO -eq 0 ]] && cmd+=(--dry-run)
  "${cmd[@]}"
  echo
done

if [[ $GO -eq 1 ]]; then
  # The checkpoints are chmod a-w and the uploader reads them in place, so nothing should
  # have changed -- but this is the cheap proof rather than the assumption. Scoped to what
  # was actually published: hashing all of pi0_full is 16 GB off Lustre and takes minutes,
  # where one published checkpoint is seconds (lora) or about a minute (full).
  echo "======== verifying published checkpoints are untouched ========"
  declare -A PREFIX=( [lora]="pi0_libero/adapter_best/" [full]="pi0_full/model_best/" )
  MANIFEST=$(mktemp); trap 'rm -f "$MANIFEST"' EXIT
  for t in "${TARGETS[@]}"; do grep -F "  ${PREFIX[$t]}" "$REPO_ROOT/checkpoints.sha256"; done > "$MANIFEST"
  if [[ ! -s "$MANIFEST" ]]; then
    echo "WARNING: no manifest entries matched; skipping verification" >&2
  else
    ( cd "$CK" && sha256sum -c "$MANIFEST" --quiet ) \
      && echo "OK: $(wc -l < "$MANIFEST") file(s) match checkpoints.sha256"
  fi
  echo
  echo "NOTE: visibility is set only when a repo is CREATED. Re-running with --public will"
  echo "      NOT flip an existing private repo; use HfApi.update_repo_settings for that."
  for t in "${TARGETS[@]}"; do echo "  https://huggingface.co/$HF_USER/${NAME[$t]}"; done
fi
