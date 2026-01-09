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

HEAD_IP="${HEAD_IP:-192.168.10.14}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_METRICS_EXPORT_PORT="${RAY_METRICS_EXPORT_PORT:-18080}"
RAY_MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-10002}"
RAY_MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-10100}"

PROMETHEUS_HOST_PORT="${PROMETHEUS_HOST_PORT:-9091}"
GRAFANA_HOST_PORT="${GRAFANA_HOST_PORT:-3000}"

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

start_prometheus() {
  local prom_cfg="/tmp/ray/session_latest/metrics/prometheus/prometheus.yml"

  # Pick a free port before starting Ray so Ray can generate Grafana datasource config correctly.
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${PROMETHEUS_HOST_PORT}\$"; then
    echo "Prometheus port ${PROMETHEUS_HOST_PORT} is in use; searching for a free port..."
    PROMETHEUS_HOST_PORT="$(find_free_port 9091 9191)"
    echo "Using Prometheus port ${PROMETHEUS_HOST_PORT}"
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

  echo "Starting Prometheus: ${prom_bin} (port=${PROMETHEUS_HOST_PORT})"
  echo "  config: ${prom_cfg}"
  echo "  log   : ${PROM_LOG_FILE}"
  nohup "${prom_bin}" \
    --config.file="${prom_cfg}" \
    --web.listen-address="0.0.0.0:${PROMETHEUS_HOST_PORT}" \
    --storage.tsdb.path="${PROM_DATA_DIR}" \
    --web.enable-lifecycle \
    >"${PROM_LOG_FILE}" 2>&1 &
  echo $! > "${PROM_PID_FILE}"
}

start_grafana() {
  # Pick a free port before starting Ray so Ray can expose the correct iframe host.
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":${GRAFANA_HOST_PORT}\$"; then
    echo "Grafana port ${GRAFANA_HOST_PORT} is in use; searching for a free port..."
    GRAFANA_HOST_PORT="$(find_free_port 3000 3099)"
    echo "Using Grafana port ${GRAFANA_HOST_PORT}"
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker not found. Install Docker, or start Grafana manually with Ray's generated config:" >&2
    echo "  grafana.ini: /tmp/ray/session_latest/metrics/grafana/grafana.ini" >&2
    echo "  provisioning: /tmp/ray/session_latest/metrics/grafana/provisioning/" >&2
    return 1
  fi

  # Restart container every time to pick up the latest session configs.
  (docker rm -f "${GRAFANA_CONTAINER_NAME}" >/dev/null 2>&1 || true)

  echo "Starting Grafana (docker) on host port ${GRAFANA_HOST_PORT}..."
  echo "  (using host networking so Prometheus at ${RAY_PROMETHEUS_HOST} is reachable from Grafana)"

  # IMPORTANT:
  # - Ray writes Grafana provisioning under /tmp/ray/session_latest/metrics/grafana/provisioning/
  # - Grafana docker image defaults GF_PATHS_PROVISIONING to /etc/grafana/provisioning, which would
  #   ignore Ray's generated dashboards unless we override it.
  # - We use --network host so the provisioned datasource URL (127.0.0.1:<prom_port>) works.
  docker run -d --name "${GRAFANA_CONTAINER_NAME}" --restart unless-stopped \
    --network host \
    -v /tmp/ray:/tmp/ray \
    -e GF_PATHS_CONFIG=/tmp/ray/session_latest/metrics/grafana/grafana.ini \
    -e GF_PATHS_PROVISIONING=/tmp/ray/session_latest/metrics/grafana/provisioning \
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

# --- 2) Start Ray head ---
ray stop -f
ray start --head \
  --node-ip-address="${HEAD_IP}" --port="${RAY_GCS_PORT}" \
  --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}" \
  --metrics-export-port="${RAY_METRICS_EXPORT_PORT}" \
  --num-gpus=1 \
  --min-worker-port="${RAY_MIN_WORKER_PORT}" --max-worker-port="${RAY_MAX_WORKER_PORT}"

# --- 3) Wait for Ray to generate metrics configs ---
wait_for_file "/tmp/ray/session_latest/metrics/prometheus/prometheus.yml" 60
wait_for_file "/tmp/ray/session_latest/metrics/grafana/grafana.ini" 60

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
healthcheck_http "${RAY_PROMETHEUS_HOST}/-/healthy"
healthcheck_http "${RAY_GRAFANA_HOST}/api/health"
healthcheck_http "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/prometheus_health"
healthcheck_http "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/grafana_health"

