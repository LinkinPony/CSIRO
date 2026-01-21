#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for a Ray WORKER node (join an existing Ray head).
#
# Usage (on worker machine 192.168.199.241):
#   bash run_ray_worker.sh
#
# Override via env vars if needed:
#   HEAD_IP=... WORKER_IP=... bash run_ray_worker.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Ensure Ray worker processes can import this repo's `src` package.
# (Ray tasks/actors may run with a different CWD/sys.path than this shell.)
export CSIRO_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HEAD_IP="${HEAD_IP:-192.168.10.14}"
HEAD_PORT="${HEAD_PORT:-6379}"

# IMPORTANT: Set this to the worker machine's reachable IP on the Ray network.
WORKER_IP="${WORKER_IP:-192.168.199.241}"

RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-10100}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-18080}"

NUM_GPUS="${NUM_GPUS:-1}"

echo "Starting Ray worker:"
echo "  head    : ${HEAD_IP}:${HEAD_PORT}"
echo "  worker  : ${WORKER_IP}"
echo "  gpus    : ${NUM_GPUS}"
echo "  metrics : ${RAY_METRICS_EXPORT_PORT}"
echo "  ports   : ${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
echo ""

ray stop -f || true

ray start \
  --address="${HEAD_IP}:${HEAD_PORT}" \
  --node-ip-address="${WORKER_IP}" \
  --num-gpus="${NUM_GPUS}" \
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" --max-worker-port="${RAY_MAX_WORKER_PORT}"

echo ""
echo "Worker joined. Check cluster status from head:"
echo "  ray status"


