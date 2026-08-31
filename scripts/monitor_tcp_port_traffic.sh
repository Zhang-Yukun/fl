#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
TCPDUMP_BIN="${TCPDUMP_BIN:-tcpdump}"
INTERFACE="${INTERFACE:-any}"
PORT=""
OUTPUT_DIR=""
RAW_LOG_NAME="${RAW_LOG_NAME:-grpc_port_traffic.tcpdump.log}"
ERR_LOG_NAME="${ERR_LOG_NAME:-grpc_port_traffic.tcpdump.stderr.log}"
SUMMARY_NAME="${SUMMARY_NAME:-grpc_port_traffic.summary.json}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
TCPDUMP_PID=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/monitor_tcp_port_traffic.sh --port 50051 --output-dir outputs/monitor

Optional flags:
  --port PORT
  --output-dir DIR
  --interface IFACE
  --tcpdump-bin PATH
  --python-bin PATH
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --interface)
      INTERFACE="$2"
      shift 2
      ;;
    --tcpdump-bin)
      TCPDUMP_BIN="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${PORT}" || -z "${OUTPUT_DIR}" ]]; then
  usage >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
RAW_LOG="${OUTPUT_DIR}/${RAW_LOG_NAME}"
ERR_LOG="${OUTPUT_DIR}/${ERR_LOG_NAME}"
SUMMARY_PATH="${OUTPUT_DIR}/${SUMMARY_NAME}"

summarize_capture() {
  "${PYTHON_BIN}" - "$PORT" "$RAW_LOG" "$ERR_LOG" "$SUMMARY_PATH" <<'PY_SUMMARY'
import json
import re
import sys
from pathlib import Path

port = int(sys.argv[1])
raw_log = Path(sys.argv[2])
err_log = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
pattern = re.compile(r"^(?P<ts>\S+)\s+(?:(?:\S+)\s+){0,3}(?:IP6?|IP)\s+(?P<src>\S+)\s+>\s+(?P<dst>\S+):.*\slength\s+(?P<length>\d+)\s*$")

def endpoint_port(endpoint: str):
    endpoint = endpoint.rstrip(':')
    if '.' not in endpoint:
        return None
    suffix = endpoint.rsplit('.', 1)[-1]
    return int(suffix) if suffix.isdigit() else None

sent = 0
received = 0
sent_packets = 0
received_packets = 0
ignored_lines = 0
matched_lines = 0
lines = raw_log.read_text(encoding='utf-8', errors='replace').splitlines() if raw_log.exists() else []
for line in lines:
    match = pattern.match(line.strip())
    if not match:
        ignored_lines += 1
        continue
    matched_lines += 1
    src_port = endpoint_port(match.group('src'))
    dst_port = endpoint_port(match.group('dst'))
    length = int(match.group('length'))
    if src_port == port and dst_port != port:
        sent += length
        sent_packets += 1
    elif dst_port == port and src_port != port:
        received += length
        received_packets += 1
    else:
        ignored_lines += 1
payload = {
    'port': port,
    'sent_payload_bytes': sent,
    'received_payload_bytes': received,
    'total_payload_bytes': sent + received,
    'sent_packets': sent_packets,
    'received_packets': received_packets,
    'matched_lines': matched_lines,
    'ignored_lines': ignored_lines,
    'raw_log_path': str(raw_log),
    'stderr_log_path': str(err_log),
    'tcpdump_stderr': err_log.read_text(encoding='utf-8', errors='replace') if err_log.exists() else '',
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
PY_SUMMARY
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${TCPDUMP_PID}" ]] && kill -0 "${TCPDUMP_PID}" 2>/dev/null; then
    kill "${TCPDUMP_PID}" 2>/dev/null || true
    wait "${TCPDUMP_PID}" 2>/dev/null || true
  fi
  summarize_capture
  exit "${status}"
}

trap cleanup EXIT HUP INT TERM

echo "[monitor] starting tcpdump port=${PORT} interface=${INTERFACE} output_dir=${OUTPUT_DIR}"
"${TCPDUMP_BIN}" -n -l -tt -i "${INTERFACE}" "tcp port ${PORT}" > "${RAW_LOG}" 2> "${ERR_LOG}" &
TCPDUMP_PID=$!

while kill -0 "${TCPDUMP_PID}" 2>/dev/null; do
  sleep "${SLEEP_SECONDS}"
done
wait "${TCPDUMP_PID}" || true
