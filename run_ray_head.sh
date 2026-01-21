#!/usr/bin/env bash
set -euo pipefail

# One-click launcher for:
# - Ray head node
# - Prometheus (for Ray Dashboard time-series metrics)
# - Grafana (for embedded dashboards inside Ray Dashboard)
#
# Notes:
# - We intentionally run Prometheus on a non-default port (9091 by default),
#   because 9090 is often occupied by other services.
# - Ray's built-in `ray metrics launch-prometheus` can't change the listen port,
#   so we start Prometheus manually using the downloaded binary.
# - Grafana is started via Docker (recommended). If you don't have Docker,
#   install it or run Grafana manually.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

# Ensure Ray head/worker processes can import this repo's `src` package.
# (Ray tasks/actors may run with a different CWD/sys.path than this shell.)
export CSIRO_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

HEAD_IP="${HEAD_IP:-192.168.10.14}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-18080}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-10100}"

# Ray writes its session state under a temp dir (defaults to /tmp/ray).
# If this script was ever run with sudo, /tmp/ray can become root-owned and break
# subsequent non-root runs with: PermissionError: [Errno 13] Permission denied: '/tmp/ray/session_*'
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray}"
RAY_TEMP_DIR_FALLBACK="${RAY_TEMP_DIR_FALLBACK:-/tmp/ray_${USER}}"

PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9091}"
GRAFANA_HOST_PORT="${GRAFANA_HOST_PORT:-3000}"

# Ray's built-in Grafana dashboards filter on `ray_io_cluster=~"$Cluster"` by default.
# On bare-metal Ray (non-KubeRay), the `ray_io_cluster` label is usually absent, which
# makes the dashboards look empty even though metrics exist. We fix this by injecting
# a constant `ray_io_cluster` label at scrape time.
RAY_IO_CLUSTER_LABEL_VALUE="${RAY_IO_CLUSTER_LABEL_VALUE:-local}"

PROM_PID_FILE="${PROM_PID_FILE:-/tmp/ray_prometheus.pid}"
PROM_LOG_FILE="${PROM_LOG_FILE:-/tmp/ray_prometheus.log}"
PROM_DATA_DIR="${PROM_DATA_DIR:-/tmp/ray_prometheus_data}"

GRAFANA_CONTAINER_NAME="${GRAFANA_CONTAINER_NAME:-ray-grafana}"

find_free_port() {
  local start_port="$1"
  local end_port="$2"
  local p
  for ((p=start_port; p<=end_port; p++)); do
    if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${p}\$"; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

wait_for_file() {
  local path="$1"
  local timeout_s="${2:-30}"
  local t=0
  while [[ ! -f "${path}" ]]; do
    if (( t >= timeout_s )); then
      echo "ERROR: timeout waiting for file: ${path}" >&2
      return 1
    fi
    sleep 1
    t=$((t+1))
  done
}

stop_prometheus_if_running() {
  if [[ -f "${PROM_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${PROM_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping Prometheus (pid=${pid})..."
      kill "${pid}" || true
      sleep 1
      if kill -0 "${pid}" 2>/dev/null; then
        kill -9 "${pid}" || true
      fi
    fi
    rm -f "${PROM_PID_FILE}" || true
  fi
}

ensure_ray_temp_dir() {
  mkdir -p "${RAY_TEMP_DIR}" 2>/dev/null || true

  if [[ -w "${RAY_TEMP_DIR}" ]]; then
    return 0
  fi

  echo "WARN: Ray temp dir is not writable: ${RAY_TEMP_DIR}" >&2
  echo "  (common cause: /tmp/ray is root-owned from a previous sudo run)" >&2
  echo "  Falling back to: ${RAY_TEMP_DIR_FALLBACK}" >&2

  RAY_TEMP_DIR="${RAY_TEMP_DIR_FALLBACK}"
  mkdir -p "${RAY_TEMP_DIR}"
  chmod 700 "${RAY_TEMP_DIR}" 2>/dev/null || true

  if [[ ! -w "${RAY_TEMP_DIR}" ]]; then
    echo "ERROR: Ray temp dir still not writable: ${RAY_TEMP_DIR}" >&2
    echo "  Fix: choose a writable directory, e.g. export RAY_TEMP_DIR=\$HOME/.ray_tmp" >&2
    return 1
  fi
}

start_prometheus() {
  local prom_cfg="${RAY_TEMP_DIR}/session_latest/metrics/prometheus/prometheus.yml"
  local prom_cfg_patched="${RAY_TEMP_DIR}/session_latest/metrics/prometheus/prometheus.patched.yml"

  # IMPORTANT: PROMETHEUS_HOST_PORT is selected *before* starting Ray and exported via
  # RAY_PROMETHEUS_HOST. Do NOT change the port here, otherwise Ray/Grafana will point
  # to the wrong address. If the port is still busy, fail and ask the user to rerun.
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PROMETHEUS_HOST_PORT}\$"; then
    echo "ERROR: Prometheus port ${PROMETHEUS_HOST_PORT} is in use (after stopping previous Prometheus)." >&2
    echo "  Hint: set PROMETHEUS_HOST_PORT=<free_port> and rerun ./run_ray_head.sh" >&2
    return 1
  fi

  # Best-effort: find a Prometheus binary. Prefer repo-local extracted versions.
  local prom_bin="${PROMETHEUS_BIN:-}"
  if [[ -z "${prom_bin}" ]]; then
    local candidates=()
    # shellcheck disable=SC2206
    candidates=( ${REPO_ROOT}/prometheus-*.linux-amd64/prometheus )
    if [[ -e "${candidates[0]:-}" ]]; then
      prom_bin="$(printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1)"
    elif command -v prometheus >/dev/null 2>&1; then
      prom_bin="$(command -v prometheus)"
    fi
  fi

  if [[ -z "${prom_bin}" ]] || [[ ! -x "${prom_bin}" ]]; then
    echo "Prometheus binary not found. Attempting to download via \`ray metrics launch-prometheus\` (may fail to start on :9090; that's OK)..."
    (ray metrics launch-prometheus || true) >/tmp/ray_metrics_launch_prometheus.log 2>&1 || true
    (ray metrics shutdown-prometheus || true) >/dev/null 2>&1 || true

    # Retry discovery
    local candidates=()
    # shellcheck disable=SC2206
    candidates=( ${REPO_ROOT}/prometheus-*.linux-amd64/prometheus )
    if [[ -e "${candidates[0]:-}" ]]; then
      prom_bin="$(printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1)"
    fi
  fi

  if [[ -z "${prom_bin}" ]] || [[ ! -x "${prom_bin}" ]]; then
    echo "ERROR: Prometheus binary still not found. You can manually install it, or re-run:" >&2
    echo "  cd ${REPO_ROOT} && ray metrics launch-prometheus" >&2
    return 1
  fi

  stop_prometheus_if_running
  mkdir -p "${PROM_DATA_DIR}"

  # Patch Prometheus config to inject `ray_io_cluster` label so Ray's Grafana dashboards work.
  # We write a separate file to avoid mutating Ray-managed configs.
  PROM_CFG="${prom_cfg}" \
  PROM_CFG_PATCHED="${prom_cfg_patched}" \
  RAY_IO_CLUSTER_LABEL_VALUE="${RAY_IO_CLUSTER_LABEL_VALUE}" \
  python - <<'PY'
import json
import os
import re
from pathlib import Path

src = Path(os.environ["PROM_CFG"])
dst = Path(os.environ["PROM_CFG_PATCHED"])
value = os.environ.get("RAY_IO_CLUSTER_LABEL_VALUE", "local")

text = src.read_text(encoding="utf-8")

# If the relabel already exists, keep the config unchanged.
if re.search(r"\btarget_label:\s*ray_io_cluster\b", text):
    dst.write_text(text, encoding="utf-8")
    print(f"Wrote patched Prometheus config (already present): {dst} (ray_io_cluster={value!r})")
    raise SystemExit(0)

lines = text.splitlines()
out = []
in_ray_job = False
job_indent = None
inserted = False

ray_job_re = re.compile(r"^(\s*)-\s*job_name:\s*['\"]?ray['\"]?\s*$")
job_start_re = re.compile(r"^(\s*)-\s*job_name:\s*")

def _emit_relabel_block(indent_spaces: int) -> list[str]:
    indent = " " * indent_spaces
    # Use JSON quoting to safely produce a YAML-compatible double-quoted string.
    q = json.dumps(str(value))
    return [
        f"{indent}relabel_configs:",
        f"{indent}- target_label: ray_io_cluster",
        f"{indent}  replacement: {q}",
    ]

for line in lines:
    m_ray = ray_job_re.match(line)
    if m_ray:
        in_ray_job = True
        job_indent = len(m_ray.group(1))
        out.append(line)
        continue

    m_job = job_start_re.match(line)
    if in_ray_job and not inserted and m_job and len(m_job.group(1)) == (job_indent or 0):
        # We are about to enter the next scrape_config job. Insert our relabel block
        # at the end of the ray job (2 spaces deeper than the job list item).
        out.extend(_emit_relabel_block((job_indent or 0) + 2))
        inserted = True
        in_ray_job = False
        job_indent = None
        out.append(line)
        continue

    out.append(line)

# If the ray job was the last job in the file, append relabel block at EOF.
if in_ray_job and not inserted:
    out.extend(_emit_relabel_block((job_indent or 0) + 2))
    inserted = True

dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"Wrote patched Prometheus config: {dst} (ray_io_cluster={value!r}, inserted={inserted})")
PY

  echo "Starting Prometheus: ${prom_bin} (port=${PROMETHEUS_HOST_PORT})"
  echo "  config: ${prom_cfg_patched}"
  echo "  log   : ${PROM_LOG_FILE}"
  nohup "${prom_bin}" \
    --config.file="${prom_cfg_patched}" \
    --web.listen-address="0.0.0.0:${PROMETHEUS_HOST_PORT}" \
    --storage.tsdb.path="${PROM_DATA_DIR}" \
    --web.enable-lifecycle \
    >"${PROM_LOG_FILE}" 2>&1 &
  echo $! > "${PROM_PID_FILE}"
}

start_grafana() {
  # IMPORTANT: GRAFANA_HOST_PORT is selected *before* starting Ray and exported via
  # RAY_GRAFANA_HOST / RAY_GRAFANA_IFRAME_HOST. Do NOT change the port here.
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${GRAFANA_HOST_PORT}\$"; then
    echo "ERROR: Grafana port ${GRAFANA_HOST_PORT} is in use." >&2
    echo "  Hint: set GRAFANA_HOST_PORT=<free_port> and rerun ./run_ray_head.sh" >&2
    return 1
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker not found. Install Docker, or start Grafana manually with Ray's generated config:" >&2
    echo "  grafana.ini: ${RAY_TEMP_DIR}/session_latest/metrics/grafana/grafana.ini" >&2
    echo "  provisioning: ${RAY_TEMP_DIR}/session_latest/metrics/grafana/provisioning/" >&2
    return 1
  fi

  # Ensure we can actually talk to the Docker daemon (common failure: user not in docker group
  # in the current shell session; requires re-login or `newgrp docker`).
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker is installed but not accessible (permission denied to /var/run/docker.sock)." >&2
    echo "  Fix (pick one):" >&2
    echo "    - Open a NEW shell / re-login so group membership refreshes (id -nG should include 'docker')" >&2
    echo "    - Run: newgrp docker" >&2
    echo "    - Or run with sudo (if configured): sudo ./run_ray_head.sh" >&2
    return 1
  fi

  # Restart container every time to pick up the latest session configs.
  (docker rm -f "${GRAFANA_CONTAINER_NAME}" >/dev/null 2>&1 || true)

  echo "Starting Grafana (docker) on host port ${GRAFANA_HOST_PORT}..."
  echo "  (using host networking so Prometheus at ${RAY_PROMETHEUS_HOST} is reachable from Grafana)"

  # IMPORTANT:
  # - Ray writes Grafana provisioning under ${RAY_TEMP_DIR}/session_latest/metrics/grafana/provisioning/
  # - Grafana docker image defaults GF_PATHS_PROVISIONING to /etc/grafana/provisioning, which would
  #   ignore Ray's generated dashboards unless we override it.
  # - We use --network host so the provisioned datasource URL (127.0.0.1:<prom_port>) works.
  # - We mount Ray's temp dir into the container at the SAME absolute path:
  #   Ray creates session_latest as an absolute symlink (e.g. /tmp/ray_dl/session_latest -> /tmp/ray_dl/session_xxx),
  #   so mounting to a different container path would break the symlink.
  docker run -d --name "${GRAFANA_CONTAINER_NAME}" --restart unless-stopped \
    --network host \
    -v "${RAY_TEMP_DIR}:${RAY_TEMP_DIR}" \
    -e "GF_PATHS_CONFIG=${RAY_TEMP_DIR}/session_latest/metrics/grafana/grafana.ini" \
    -e "GF_PATHS_PROVISIONING=${RAY_TEMP_DIR}/session_latest/metrics/grafana/provisioning" \
    -e GF_SERVER_HTTP_PORT="${GRAFANA_HOST_PORT}" \
    grafana/grafana-oss:latest >/dev/null
}

healthcheck_http() {
  local url="$1"
  python - <<PY
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open("${url}", timeout=2) as r:
        print("${url}", "OK", r.status)
except Exception as e:
    print("${url}", "ERR", e)
PY
}

wait_for_http_ok() {
  local url="$1"
  local timeout_s="${2:-60}"
  local t=0
  while (( t < timeout_s )); do
    if python - <<PY >/dev/null 2>&1
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open("${url}", timeout=2) as r:
    assert 200 <= r.status < 400
PY
    then
      return 0
    fi
    sleep 1
    t=$((t+1))
  done
  return 1
}

# --- 0) Stop previous monitoring processes (best-effort) ---
# Stop Prometheus from a previous run so we can reuse the default port if available.
stop_prometheus_if_running || true
# Stop previous Grafana container (best-effort); ignore errors (e.g., no docker permission yet).
(docker rm -f "${GRAFANA_CONTAINER_NAME}" >/dev/null 2>&1 || true)

# --- 1) Choose ports + export env vars BEFORE starting Ray (required for dashboard detection) ---
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PROMETHEUS_HOST_PORT}\$"; then
  PROMETHEUS_HOST_PORT="$(find_free_port 9091 9191)"
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${GRAFANA_HOST_PORT}\$"; then
  GRAFANA_HOST_PORT="$(find_free_port 3000 3099)"
fi

export RAY_PROMETHEUS_HOST="http://127.0.0.1:${PROMETHEUS_HOST_PORT}"
export RAY_GRAFANA_HOST="http://127.0.0.1:${GRAFANA_HOST_PORT}"
export RAY_GRAFANA_IFRAME_HOST="http://${HEAD_IP}:${GRAFANA_HOST_PORT}"

echo "Ray metrics integration:"
echo "  RAY_PROMETHEUS_HOST=${RAY_PROMETHEUS_HOST}"
echo "  RAY_GRAFANA_HOST=${RAY_GRAFANA_HOST}"
echo "  RAY_GRAFANA_IFRAME_HOST=${RAY_GRAFANA_IFRAME_HOST}"
echo "  ray_io_cluster label=${RAY_IO_CLUSTER_LABEL_VALUE}"

ensure_ray_temp_dir
echo "  ray temp dir=${RAY_TEMP_DIR}"

# --- 2) Start Ray head ---
ray stop -f
ray start --head \
  --node-ip-address="${HEAD_IP}" --port="${RAY_GCS_PORT}" \
  --temp-dir="${RAY_TEMP_DIR}" \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --num-gpus=1 \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" --max-worker-port="${RAY_MAX_WORKER_PORT}"

# --- 3) Wait for Ray to generate metrics configs ---
wait_for_file "${RAY_TEMP_DIR}/session_latest/metrics/prometheus/prometheus.yml" 60
wait_for_file "${RAY_TEMP_DIR}/session_latest/metrics/grafana/grafana.ini" 60

# --- 4) Start Prometheus + Grafana ---
start_prometheus
start_grafana

echo ""
echo "URLs:"
echo "  Ray Dashboard : http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
echo "  Prometheus    : http://${HEAD_IP}:${PROMETHEUS_HOST_PORT}"
echo "  Grafana       : http://${HEAD_IP}:${GRAFANA_HOST_PORT}"
echo ""
echo "Healthchecks:"
wait_for_http_ok "${RAY_PROMETHEUS_HOST}/-/healthy" 30 || true
wait_for_http_ok "${RAY_GRAFANA_HOST}/api/health" 60 || true
wait_for_http_ok "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/prometheus_health" 30 || true
wait_for_http_ok "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/grafana_health" 60 || true
healthcheck_http "${RAY_PROMETHEUS_HOST}/-/healthy"
healthcheck_http "${RAY_GRAFANA_HOST}/api/health"
healthcheck_http "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/prometheus_health"
healthcheck_http "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/grafana_health"

