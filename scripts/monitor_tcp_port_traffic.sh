#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
TSHARK_BIN="${TSHARK_BIN:-tshark}"
INTERFACE="${INTERFACE:-any}"
PORT=""
OUTPUT_DIR=""
RAW_PCAP_NAME="${RAW_PCAP_NAME:-grpc_port_traffic.pcap}"
ERR_LOG_NAME="${ERR_LOG_NAME:-grpc_port_traffic.tshark.stderr.log}"
SUMMARY_NAME="${SUMMARY_NAME:-grpc_port_traffic.summary.json}"
SLEEP_SECONDS="${SLEEP_SECONDS:-1}"
CAPTURE_PID=""

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/monitor_tcp_port_traffic.sh --port 50051 --output-dir outputs/monitor

Optional flags:
  --port PORT
  --output-dir DIR
  --interface IFACE
  --tshark-bin PATH
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
    --tshark-bin)
      TSHARK_BIN="$2"
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
RAW_PCAP="${OUTPUT_DIR}/${RAW_PCAP_NAME}"
ERR_LOG="${OUTPUT_DIR}/${ERR_LOG_NAME}"
SUMMARY_PATH="${OUTPUT_DIR}/${SUMMARY_NAME}"

summarize_capture() {
  "${PYTHON_BIN}" - "$TSHARK_BIN" "$PORT" "$RAW_PCAP" "$ERR_LOG" "$SUMMARY_PATH" <<'PY_SUMMARY'
import json
import subprocess
import sys
from pathlib import Path

tshark_bin = sys.argv[1]
port = int(sys.argv[2])
pcap_path = Path(sys.argv[3])
err_log = Path(sys.argv[4])
summary_path = Path(sys.argv[5])

sent = 0
received = 0
sent_payload = 0
received_payload = 0
sent_packets = 0
received_packets = 0
parsed_packets = 0
skipped_packets = 0

if pcap_path.exists() and pcap_path.stat().st_size > 0:
    command = [
        tshark_bin,
        '-r',
        str(pcap_path),
        '-T',
        'fields',
        '-E',
        'separator=\t',
        '-e',
        'tcp.srcport',
        '-e',
        'tcp.dstport',
        '-e',
        'frame.len',
        '-e',
        'tcp.len',
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        parts = line.rstrip('\n').split('\t')
        if len(parts) != 4:
            skipped_packets += 1
            continue
        src_port, dst_port, frame_len, tcp_len = parts
        if not frame_len.isdigit():
            skipped_packets += 1
            continue
        parsed_packets += 1
        frame_len_value = int(frame_len)
        tcp_len_value = int(tcp_len) if tcp_len.isdigit() else 0
        src_port_value = int(src_port) if src_port.isdigit() else None
        dst_port_value = int(dst_port) if dst_port.isdigit() else None
        if src_port_value == port and dst_port_value != port:
            sent += frame_len_value
            sent_payload += tcp_len_value
            sent_packets += 1
        elif dst_port_value == port and src_port_value != port:
            received += frame_len_value
            received_payload += tcp_len_value
            received_packets += 1
        else:
            skipped_packets += 1
payload = {
    'port': port,
    'sent_bytes': sent,
    'received_bytes': received,
    'total_bytes': sent + received,
    'sent_tcp_payload_bytes': sent_payload,
    'received_tcp_payload_bytes': received_payload,
    'total_tcp_payload_bytes': sent_payload + received_payload,
    'sent_packets': sent_packets,
    'received_packets': received_packets,
    'parsed_packets': parsed_packets,
    'skipped_packets': skipped_packets,
    'pcap_path': str(pcap_path),
    'stderr_log_path': str(err_log),
    'capture_stderr': err_log.read_text(encoding='utf-8', errors='replace') if err_log.exists() else '',
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
PY_SUMMARY
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${CAPTURE_PID}" ]] && kill -0 "${CAPTURE_PID}" 2>/dev/null; then
    kill "${CAPTURE_PID}" 2>/dev/null || true
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
  summarize_capture
  exit "${status}"
}

trap cleanup EXIT HUP INT TERM

echo "[monitor] starting tshark capture port=${PORT} interface=${INTERFACE} output_dir=${OUTPUT_DIR}"
"${TSHARK_BIN}" -n -i "${INTERFACE}" -f "tcp port ${PORT}" -s 0 -w "${RAW_PCAP}" > /dev/null 2> "${ERR_LOG}" &
CAPTURE_PID=$!

while kill -0 "${CAPTURE_PID}" 2>/dev/null; do
  sleep "${SLEEP_SECONDS}"
done
wait "${CAPTURE_PID}" || true
