#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
TCPDUMP_BIN="${TCPDUMP_BIN:-tcpdump}"
TSHARK_BIN="${TSHARK_BIN:-tshark}"
INTERFACE="${INTERFACE:-any}"
PORT=""
SERVER_IP=""
OUTPUT_DIR=""
RAW_PCAP_NAME="${RAW_PCAP_NAME:-grpc_port_traffic.pcap}"
ERR_LOG_NAME="${ERR_LOG_NAME:-grpc_port_traffic.capture.stderr.log}"
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
  --server-ip IP
  --interface IFACE
  --tcpdump-bin PATH
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
    --server-ip)
      SERVER_IP="$2"
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
  "${PYTHON_BIN}" - "$TSHARK_BIN" "$PORT" "$SERVER_IP" "$RAW_PCAP" "$ERR_LOG" "$SUMMARY_PATH" <<'PY_SUMMARY'
import json
import socket
import subprocess
import sys
from pathlib import Path

tshark_bin = sys.argv[1]
port = int(sys.argv[2])
server_ip = sys.argv[3].strip()
pcap_path = Path(sys.argv[4])
err_log = Path(sys.argv[5])
summary_path = Path(sys.argv[6])

sent = 0
received = 0
sent_payload = 0
received_payload = 0
sent_packets = 0
received_packets = 0
parsed_packets = 0
skipped_packets = 0
parse_error = None


def _collect_local_ips() -> set[str]:
    values = {'127.0.0.1', '::1'}
    commands = (
        ['ip', '-o', 'addr', 'show'],
        ['hostname', '-I'],
    )
    for command in commands:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if command[:3] == ['ip', '-o', 'addr']:
            for line in completed.stdout.splitlines():
                parts = line.split()
                if len(parts) < 4 or parts[2] not in {'inet', 'inet6'}:
                    continue
                values.add(parts[3].split('/', 1)[0])
        else:
            for item in completed.stdout.split():
                if item:
                    values.add(item.strip())
    for host in (socket.gethostname(), socket.getfqdn(), 'localhost'):
        try:
            for family, _kind, _proto, _canonname, sockaddr in socket.getaddrinfo(host, None):
                if family in (socket.AF_INET, socket.AF_INET6) and sockaddr:
                    values.add(str(sockaddr[0]))
        except socket.gaierror:
            continue
    return {value for value in values if value}


local_ips = {server_ip} if server_ip else _collect_local_ips()

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
        'ip.src',
        '-e',
        'ip.dst',
        '-e',
        'ipv6.src',
        '-e',
        'ipv6.dst',
        '-e',
        'tcp.srcport',
        '-e',
        'tcp.dstport',
        '-e',
        'frame.len',
        '-e',
        'tcp.len',
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        parse_error = exc.stderr or str(exc)
    else:
        for line in completed.stdout.splitlines():
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 8:
                skipped_packets += 1
                continue
            ip_src, ip_dst, ipv6_src, ipv6_dst, src_port, dst_port, frame_len, tcp_len = parts
            if not frame_len.isdigit():
                skipped_packets += 1
                continue
            parsed_packets += 1
            src_ip = ip_src or ipv6_src
            dst_ip = ip_dst or ipv6_dst
            frame_len_value = int(frame_len)
            tcp_len_value = int(tcp_len) if tcp_len.isdigit() else 0
            src_port_value = int(src_port) if src_port.isdigit() else None
            dst_port_value = int(dst_port) if dst_port.isdigit() else None
            if src_port_value == port and src_ip in local_ips:
                sent += frame_len_value
                sent_payload += tcp_len_value
                sent_packets += 1
            elif dst_port_value == port and dst_ip in local_ips:
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
    'server_ip': server_ip or None,
    'local_ips': sorted(local_ips),
    'pcap_path': str(pcap_path),
    'stderr_log_path': str(err_log),
    'capture_stderr': err_log.read_text(encoding='utf-8', errors='replace') if err_log.exists() else '',
    'parse_error': parse_error,
}
summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
PY_SUMMARY
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${CAPTURE_PID}" ]] && kill -0 "${CAPTURE_PID}" 2>/dev/null; then
    kill -INT "${CAPTURE_PID}" 2>/dev/null || kill "${CAPTURE_PID}" 2>/dev/null || true
    wait "${CAPTURE_PID}" 2>/dev/null || true
  fi
  summarize_capture
  exit "${status}"
}

trap cleanup EXIT HUP INT TERM

echo "[monitor] starting tcpdump capture port=${PORT} interface=${INTERFACE} output_dir=${OUTPUT_DIR}"
"${TCPDUMP_BIN}" -n -U -i "${INTERFACE}" -s 0 -w "${RAW_PCAP}" "tcp port ${PORT}" > /dev/null 2> "${ERR_LOG}" &
CAPTURE_PID=$!

while kill -0 "${CAPTURE_PID}" 2>/dev/null; do
  sleep "${SLEEP_SECONDS}"
done
wait "${CAPTURE_PID}" || true
