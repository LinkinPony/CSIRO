#!/usr/bin/env bash
set -euo pipefail

# Configure these for your environment:
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-YOUR_WORKER_IP}"
REMOTE_USER="${REMOTE_USER:-YOUR_USER}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/CSIRO}"

# 1 => mirror deletes (dangerous if you have extra files on remote); 0 => safer incremental sync
DELETE="${DELETE:-0}"
# 1 => also sync dinov3_weights/ (large); 0 => skip
SYNC_WEIGHTS="${SYNC_WEIGHTS:-0}"
# 1 => preview only
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${LOCAL_ROOT}" ]]; then
  echo "ERROR: LOCAL_ROOT not found: ${LOCAL_ROOT}" >&2
  exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

RSYNC_BASE=(
  -avh
  --info=progress2
  --partial
  -e "ssh -o BatchMode=yes"
  --exclude='.git/'
  --exclude='**/.git/'
  --exclude='__pycache__/'
  --exclude='**/__pycache__/'
  --exclude='*.pyc'
  --exclude='.pytest_cache/'
  --exclude='.mypy_cache/'
  --exclude='.ruff_cache/'
)

if [[ "${DELETE}" == "1" ]]; then
  RSYNC_BASE+=( --delete --delete-delay )
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_BASE+=( --dry-run )
fi

# Create remote dirs
ssh -o BatchMode=yes "${REMOTE}" "mkdir -p '${REMOTE_ROOT}/third_party'"

# Sync core code/config (fast)
rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/src/"   "${REMOTE}:${REMOTE_ROOT}/src/"
rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/conf/"  "${REMOTE}:${REMOTE_ROOT}/conf/"
rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/configs/" "${REMOTE}:${REMOTE_ROOT}/configs/"
rsync "${RSYNC_BASE[@]}" \
  "${LOCAL_ROOT}/train.py" \
  "${LOCAL_ROOT}/train_hydra.py" \
  "${LOCAL_ROOT}/tune.py" \
  "${LOCAL_ROOT}/requirements.txt" \
  "${LOCAL_ROOT}/README_TRAINING.md" \
  "${REMOTE}:${REMOTE_ROOT}/"

# Sync only the third_party pieces actually needed for training
ssh -o BatchMode=yes "${REMOTE}" "mkdir -p '${REMOTE_ROOT}/third_party/dinov3'"
rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/third_party/dinov3/" "${REMOTE}:${REMOTE_ROOT}/third_party/dinov3/"

# Optional: vendored PEFT fallback (only needed if you don't have pip peft installed)
ssh -o BatchMode=yes "${REMOTE}" "mkdir -p '${REMOTE_ROOT}/third_party/peft/src'"
rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/third_party/peft/src/" "${REMOTE}:${REMOTE_ROOT}/third_party/peft/src/"

# Optional: sync backbone weights (large)
if [[ "${SYNC_WEIGHTS}" == "1" ]]; then
  ssh -o BatchMode=yes "${REMOTE}" "mkdir -p '${REMOTE_ROOT}/dinov3_weights'"
  rsync "${RSYNC_BASE[@]}" "${LOCAL_ROOT}/dinov3_weights/" "${REMOTE}:${REMOTE_ROOT}/dinov3_weights/"
fi

echo "Done."
