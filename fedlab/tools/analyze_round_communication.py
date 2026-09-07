#!/usr/bin/env python3
"""Analyze round communication accounting for federated runs.

Examples:
    Analyze existing artifacts:
        python -m fedlab.tools.analyze_round_communication \
            ../outputs/exp/rare/multi_sync/4096/ega \
            --first-n 1

    Launch one fresh grpc_sync run first, then analyze it:
        conda run -n torch_env python -m fedlab.tools.analyze_round_communication \
            --run-experiment --task rare --algorithm ega --rounds-to-run 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fedlab.modeling import build_model


@dataclass
class PacketRecord:
    timestamp: float
    sent_frame_bytes: int = 0
    received_frame_bytes: int = 0
    sent_tcp_payload_bytes: int = 0
    received_tcp_payload_bytes: int = 0
    sent_packets: int = 0
    received_packets: int = 0


TASK_CLIENT_IDS = {
    'rare': ['Nd2O3', 'CeO2', 'La2O3'],
    'mnist': ['m1', 'm2', 'm3'],
    'cifar10': ['c1', 'c2', 'c3'],
}

TASK_CONFIG_PATHS = {
    'rare': {
        'fedavg': 'configs/rare/fedavg.yaml',
        'topk': 'configs/rare/topk.yaml',
        'ega': 'configs/rare/ega.yaml',
    },
    'mnist': {
        'fedavg': 'configs/mnist/fedavg.yaml',
        'topk': 'configs/mnist/topk.yaml',
        'ega': 'configs/mnist/ega.yaml',
    },
    'cifar10': {
        'fedavg': 'configs/cifar10/fedavg.yaml',
        'topk': 'configs/cifar10/topk.yaml',
        'ega': 'configs/cifar10/ega.yaml',
    },
}

ALGORITHM_LABELS = {
    'fedavg': 'FedAvg 稠密更新',
    'sparse_fedavg': 'Top-K 稀疏更新',
    'topk': 'Top-K 稀疏更新',
    'randomk_fedavg': 'Random-K 稀疏更新',
    'ega_fedavg': 'EGA 编码更新',
    'ega': 'EGA 编码更新',
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _load_config_artifact(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
    else:
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config artifact must be a mapping: {path}')
    return data


def _discover_config_path(run_dir: Path) -> Path | None:
    for name in ('config.yaml', 'config.json'):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _discover_monitor_summary(run_dir: Path) -> Path | None:
    candidates = list(run_dir.rglob('grpc_port_traffic.summary.json'))
    return candidates[0] if candidates else None


def _discover_pcap(run_dir: Path) -> Path | None:
    candidates = list(run_dir.rglob('grpc_port_traffic.pcap'))
    return candidates[0] if candidates else None


def _discover_run_log(run_dir: Path) -> Path | None:
    for name in ('run.log', 'server.log'):
        path = run_dir / name
        if path.exists():
            return path
    return None


def _load_round_history(run_dir: Path) -> list[dict[str, Any]]:
    metrics_path = run_dir / 'metrics.json'
    metrics = _load_json(metrics_path)
    if not isinstance(metrics, list):
        raise ValueError(f'{metrics_path} does not contain federated per-round records')
    if not metrics:
        raise ValueError(f'{metrics_path} is empty')
    return metrics


def select_round_indices(history: list[dict[str, Any]], rounds: list[int] | None, first_n: int | None) -> list[int]:
    available = [int(item['round']) for item in history]
    if rounds:
        selected = sorted(dict.fromkeys(int(value) for value in rounds))
    else:
        count = len(available) if first_n is None else max(0, min(int(first_n), len(available)))
        selected = available[:count]
    missing = [value for value in selected if value not in available]
    if missing:
        raise ValueError(f'Requested rounds are missing from metrics.json: {missing}')
    return selected


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if amount < 1024.0 or unit == 'TB':
            return f'{amount:.2f}{unit}'
        amount /= 1024.0
    return f'{amount:.2f}TB'


def _task_type(config: dict[str, Any], record: dict[str, Any]) -> str:
    return str(config.get('task', {}).get('type') or record.get('task_type') or 'unknown').lower()


def _algorithm_name(config: dict[str, Any], record: dict[str, Any]) -> str:
    return str(record.get('algorithm') or config.get('federated', {}).get('algorithm') or 'unknown')


def _round_context_hints(config: dict[str, Any], round_index: int, algorithm: str) -> list[str]:
    hints = ['server-controlled round context common to all tasks: total_clients, total_train_samples']
    if algorithm == 'ega_fedavg':
        hints.append('EGA adds ega_normalization in every round')
        if round_index == 0:
            hints.append('EGA round 0 usually also carries one-time ega_codec_payload bootstrap')
    return hints


def _parameter_upload_hints(algorithm: str, payload_kind: str) -> list[str]:
    if payload_kind == 'dense_update' or algorithm == 'fedavg':
        return ['dense client update payload visible to aggregation']
    if payload_kind in {'sparse_update', 'randomk_update'} or algorithm in {'sparse_fedavg', 'randomk_fedavg'}:
        return ['sparse trainable update payload', 'plus any dense non-trainable buffer tensors that cannot be sparsified']
    if payload_kind == 'ega_encoded_update' or algorithm == 'ega_fedavg':
        return ['EGA encoded trainable update payload (ega_payload)', 'plus dense buffer tensors for unencoded/non-trainable state']
    if payload_kind in {'qsgd_update', 'quantized_update', 'sign_update'}:
        return ['quantized client update payload visible to aggregation']
    return ['algorithm-visible client upload payload']


def _transport_download_hints() -> list[str]:
    return ['serialized envelope field round', 'serialized envelope field state', 'serialized envelope field compressed', 'serialized envelope field round_context', 'serialized envelope field stop']


def _transport_upload_hints() -> list[str]:
    return ['serialized envelope field round', 'serialized envelope field result', 'ClientResult metadata such as client_id, num_samples, loss, compressor, and communication counters']


def _safe_model_state_key_examples(config: dict[str, Any], *, max_keys: int = 10) -> dict[str, Any]:
    if not config:
        return {}
    try:
        model = build_model(config)
    except Exception as exc:
        return {'build_error': str(exc)}
    state_dict = model.state_dict()
    trainable_names = {name for name, _ in model.named_parameters()}
    buffer_names = {name for name, _ in model.named_buffers()}
    trainable_examples: list[dict[str, Any]] = []
    buffer_examples: list[dict[str, Any]] = []
    trainable_total = 0
    buffer_total = 0
    for name, tensor in state_dict.items():
        item = {
            'key': name,
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'numel': int(tensor.numel()),
            'bytes': int(tensor.numel()) * int(tensor.element_size()),
        }
        if name in trainable_names:
            trainable_total += 1
            if len(trainable_examples) < max_keys:
                trainable_examples.append(item)
        elif name in buffer_names:
            buffer_total += 1
            if len(buffer_examples) < max_keys:
                buffer_examples.append(item)
    return {
        'model_class': type(model).__name__,
        'state_dict_key_count': len(state_dict),
        'trainable_key_count': trainable_total,
        'buffer_key_count': buffer_total,
        'trainable_key_examples': trainable_examples,
        'buffer_key_examples': buffer_examples,
    }


def _common_round_context_keys(algorithm: str) -> list[str]:
    keys = ['total_clients', 'total_train_samples']
    if algorithm == 'ega_fedavg':
        keys.extend(['ega_normalization', 'ega_codec_payload(round 0 only)'])
    return keys


def _submit_update_keys(algorithm: str) -> dict[str, list[str]]:
    base_keys = ['client_id', 'num_samples', 'loss', 'aggregation_state', 'aggregation_payload_kind', 'compressor', 'parameter_upload_bytes', 'parameter_upload_parameters', 'transport_upload_bytes', 'transport_upload_overhead_bytes']
    if algorithm == 'ega_fedavg':
        return {
            'request_root_keys': ['round', 'result'],
            'result_primary_keys': [*base_keys[:4], 'ega_payload', *base_keys[4:]],
            'ega_payload_keys': ['encoded', 'shape', 'block_size', 'encoded_dim', 'quantization_level', 'encoded_dtype', 'algorithm_num_bytes', 'algorithm_num_parameters', 'observed_update_absmax'],
        }
    return {
        'request_root_keys': ['round', 'result'],
        'result_primary_keys': base_keys,
    }


def _parameter_scope_summary(algorithm: str) -> dict[str, Any]:
    common = {
        'download_formula': 'state_num_bytes(download_state) + auxiliary_payload_num_bytes(round_context)',
        'download_counts': ['下载模型 state_dict', 'round_context 中的公共标量字段'],
    }
    if algorithm == 'ega_fedavg':
        return {
            **common,
            'download_counts': ['下载模型 state_dict', 'round_context 中的 total_clients / total_train_samples', 'ega_normalization', '首轮 ega_codec_payload'],
            'upload_counts': ['ega_payload.algorithm_num_bytes', 'buffer_update 的 state_num_bytes'],
            'upload_tensor_container': ['ClientResult.ega_payload', 'ClientResult.aggregation_state'],
        }
    if algorithm in {'sparse_fedavg', 'randomk_fedavg'}:
        return {
            **common,
            'upload_counts': ['稀疏 trainable 更新', '未稀疏化 buffer state_dict'],
            'upload_tensor_container': ['ClientResult.aggregation_state'],
        }
    return {
        **common,
        'upload_counts': ['完整 aggregation_state 更新 state_dict'],
        'upload_tensor_container': ['ClientResult.aggregation_state'],
    }


def _transport_scope_summary() -> dict[str, Any]:
    return {
        'download_rpc_envelope_keys': ['round', 'state', 'compressed', 'round_context', 'stop'],
        'upload_rpc_envelope_keys': ['round', 'result'],
        'extra_counted_bytes': ['pickle 序列化后的字段名和容器结构', 'ClientResult 元数据，如 client_id、num_samples、loss、compressor、通信计数器', 'round_context 字典外壳以及 stop/compressed 等控制字段'],
    }


def _external_scope_summary() -> dict[str, Any]:
    return {
        'summary_json_keys': ['sent_bytes', 'received_bytes', 'total_bytes', 'sent_tcp_payload_bytes', 'received_tcp_payload_bytes', 'total_tcp_payload_bytes', 'sent_packets', 'received_packets', 'parsed_packets', 'skipped_packets'],
        'pcap_note': ['frame.* 更接近链路/网络侧看到的总包长', 'tcp.len 只看 TCP payload，不含 IP/TCP 头', '若发生重传、分片、抓包丢包，外部统计会偏大或偏小'],
    }


def _load_monitor_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f'Monitor summary must be a mapping: {path}')
    return payload


def _parse_log_line_timestamp(line: str) -> float | None:
    prefix = line.split(' | ', 1)[0].strip()
    try:
        return datetime.strptime(prefix, '%Y-%m-%d %H:%M:%S.%f').timestamp()
    except ValueError:
        return None


def extract_round_end_times(run_log_path: Path) -> dict[int, float]:
    payload: dict[int, float] = {}
    for line in run_log_path.read_text(encoding='utf-8', errors='replace').splitlines():
        if 'Round ' not in line or ' algorithm=' not in line:
            continue
        timestamp = _parse_log_line_timestamp(line)
        if timestamp is None:
            continue
        try:
            round_fragment = line.split('Round ', 1)[1].split(' algorithm=', 1)[0]
            round_index = int(round_fragment.strip())
        except Exception:
            continue
        payload[round_index] = timestamp
    return payload


def _classify_packet_direction(*, port: int, local_ips: set[str], src_ip: str, dst_ip: str, src_port: int | None, dst_port: int | None, frame_len: int, tcp_len: int, timestamp: float) -> PacketRecord | None:
    if src_port == port and src_ip in local_ips:
        return PacketRecord(timestamp=timestamp, sent_frame_bytes=frame_len, sent_tcp_payload_bytes=tcp_len, sent_packets=1)
    if dst_port == port and dst_ip in local_ips:
        return PacketRecord(timestamp=timestamp, received_frame_bytes=frame_len, received_tcp_payload_bytes=tcp_len, received_packets=1)
    return None


def parse_tshark_packet_records(pcap_path: Path, *, tshark_bin: str, port: int, local_ips: set[str]) -> list[PacketRecord]:
    command = [
        tshark_bin,
        '-r',
        str(pcap_path),
        '-T',
        'fields',
        '-E',
        'separator=	',
        '-e',
        'frame.time_epoch',
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
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    packet_records: list[PacketRecord] = []
    for line in completed.stdout.splitlines():
        parts = line.rstrip('\n').split('\t')
        if len(parts) != 9:
            continue
        time_epoch, ip_src, ip_dst, ipv6_src, ipv6_dst, src_port, dst_port, frame_len, tcp_len = parts
        try:
            timestamp = float(time_epoch)
            frame_len_value = int(frame_len)
        except ValueError:
            continue
        src_ip = ip_src or ipv6_src
        dst_ip = ip_dst or ipv6_dst
        src_port_value = int(src_port) if src_port.isdigit() else None
        dst_port_value = int(dst_port) if dst_port.isdigit() else None
        tcp_len_value = int(tcp_len) if tcp_len.isdigit() else 0
        record = _classify_packet_direction(port=port, local_ips=local_ips, src_ip=src_ip, dst_ip=dst_ip, src_port=src_port_value, dst_port=dst_port_value, frame_len=frame_len_value, tcp_len=tcp_len_value, timestamp=timestamp)
        if record is not None:
            packet_records.append(record)
    return packet_records


def summarize_packets_by_windows(packet_records: list[PacketRecord], round_windows: dict[int, tuple[float, float]]) -> dict[int, dict[str, Any]]:
    summary: dict[int, dict[str, Any]] = {}
    for round_index, (start_ts, end_ts) in round_windows.items():
        window_packets = [item for item in packet_records if start_ts < item.timestamp <= end_ts]
        summary[round_index] = {
            'window_start_epoch': start_ts,
            'window_end_epoch': end_ts,
            'sent_bytes': sum(item.sent_frame_bytes for item in window_packets),
            'received_bytes': sum(item.received_frame_bytes for item in window_packets),
            'total_bytes': sum(item.sent_frame_bytes + item.received_frame_bytes for item in window_packets),
            'sent_tcp_payload_bytes': sum(item.sent_tcp_payload_bytes for item in window_packets),
            'received_tcp_payload_bytes': sum(item.received_tcp_payload_bytes for item in window_packets),
            'total_tcp_payload_bytes': sum(item.sent_tcp_payload_bytes + item.received_tcp_payload_bytes for item in window_packets),
            'sent_packets': sum(item.sent_packets for item in window_packets),
            'received_packets': sum(item.received_packets for item in window_packets),
            'packet_count': len(window_packets),
        }
    return summary


def build_round_windows(selected_rounds: list[int], round_end_times: dict[int, float], packet_records: list[PacketRecord]) -> dict[int, tuple[float, float]]:
    if not selected_rounds or not packet_records:
        return {}
    if any(round_index not in round_end_times for round_index in selected_rounds):
        missing = [round_index for round_index in selected_rounds if round_index not in round_end_times]
        raise ValueError(f'run.log is missing round-end timestamps for rounds {missing}')
    base_start = min(item.timestamp for item in packet_records)
    windows: dict[int, tuple[float, float]] = {}
    for round_index in selected_rounds:
        previous_end = round_end_times.get(round_index - 1, base_start)
        windows[round_index] = (previous_end, round_end_times[round_index])
    return windows


def _build_external_capture_report(*, run_dir: Path, selected_rounds: list[int], monitor_summary_path: Path | None, pcap_path: Path | None, run_log_path: Path | None, tshark_bin: str, port: int | None, local_ips: set[str] | None) -> dict[str, Any]:
    summary_payload = _load_monitor_summary(monitor_summary_path)
    if summary_payload is None and pcap_path is None:
        return {'available': False, 'notes': ['no external monitor summary or pcap was provided']}

    resolved_port = int(summary_payload.get('port')) if summary_payload and summary_payload.get('port') is not None else port
    if resolved_port is None:
        return {'available': False, 'notes': ['external pcap parsing requires a port; pass --port or provide monitor summary.json']}
    resolved_local_ips = set(summary_payload.get('local_ips') or []) if summary_payload else set()
    if local_ips:
        resolved_local_ips.update(local_ips)
    if not resolved_local_ips and summary_payload and summary_payload.get('server_ip'):
        resolved_local_ips.add(str(summary_payload['server_ip']))

    payload: dict[str, Any] = {
        'available': True,
        'port': resolved_port,
        'local_ips': sorted(resolved_local_ips),
        'summary_path': None if monitor_summary_path is None else str(monitor_summary_path),
        'pcap_path': None if pcap_path is None else str(pcap_path),
        'run_log_path': None if run_log_path is None else str(run_log_path),
        'overall': None,
        'per_round': None,
        'notes': [],
    }
    if summary_payload is not None:
        payload['overall'] = {key: summary_payload.get(key) for key in ('sent_bytes', 'received_bytes', 'total_bytes', 'sent_tcp_payload_bytes', 'received_tcp_payload_bytes', 'total_tcp_payload_bytes', 'sent_packets', 'received_packets', 'parsed_packets', 'skipped_packets')}
        payload['notes'].append('overall external totals come from grpc_port_traffic.summary.json and count all traffic on the monitored port')
    if pcap_path is None or run_log_path is None:
        payload['notes'].append('per-round external slicing is unavailable without both run.log and grpc_port_traffic.pcap')
        return payload
    if not resolved_local_ips:
        payload['notes'].append('per-round external slicing is unavailable because local/server IPs were not provided')
        return payload
    packet_records = parse_tshark_packet_records(pcap_path, tshark_bin=tshark_bin, port=resolved_port, local_ips=resolved_local_ips)
    round_end_times = extract_round_end_times(run_log_path)
    try:
        windows = build_round_windows(selected_rounds, round_end_times, packet_records)
    except ValueError as exc:
        payload['notes'].append(f'per-round external slicing is unavailable: {exc}')
        return payload
    payload['per_round'] = summarize_packets_by_windows(packet_records, windows)
    payload['notes'].append('per-round external values are approximated by slicing pcap packets between successive server round-end log timestamps')
    payload['notes'].append('round 0 may include registration or readiness polling that happened before the first recorded round completion')
    return payload


def _build_experiment_paths(task: str, algorithm: str, output_root: Path) -> dict[str, Path]:
    run_root = output_root / task
    return {
        'base_output': output_root,
        'run_root': run_root,
        'run_dir': run_root / algorithm,
        'monitor_dir': run_root / algorithm / 'grpc_port_traffic',
    }


def _build_run_experiment_command(*, task: str, algorithm: str, rounds_to_run: int, seed: int, base_port: int, runtime_device: str, startup_wait_seconds: int, poll_seconds: float, output_root: Path) -> tuple[list[str], dict[str, str], Path]:
    paths = _build_experiment_paths(task, algorithm, output_root)
    env = os.environ.copy()
    env.update(
        {
            'PYTHON_BIN': sys.executable,
            'SUITE_SEED': str(seed),
            'RUNTIME_SEED': str(seed),
            'BASE_OUTPUT': str(paths['base_output']),
            'TASK_SET': task,
            'FEDERATED_ALGORITHMS': algorithm,
            'RUN_CENTRALIZED': 'false',
            'ROUNDS': str(rounds_to_run),
            'BASE_PORT': str(base_port),
            'RUNTIME_DEVICE': runtime_device,
            'STARTUP_WAIT_SECONDS': str(startup_wait_seconds),
            'POLL_SECONDS': str(poll_seconds),
            'MONITOR_GRPC_PORT_TRAFFIC': 'true',
            'LOSS_NAME': 'mse',
            'EGA_PRETRAIN_SEED': str(seed),
            'EGA_QUANTIZATION_SEED': str(seed),
            'QSGD_SEED': str(seed),
            'RANDOMK_SEED': str(seed),
            'QINT8_SEED': str(seed),
            'ADAPTIVE_RDP_SEED': str(seed),
        }
    )
    command = ['bash', 'scripts/run_suite.sh', '--modes', 'grpc_sync', '--tasks', task, '--algorithms', algorithm]
    return command, env, paths['run_dir']


def run_experiment_and_resolve_run_dir(*, task: str, algorithm: str, rounds_to_run: int, seed: int, base_port: int, runtime_device: str, startup_wait_seconds: int, poll_seconds: float, output_root: Path) -> Path:
    command, env, run_dir = _build_run_experiment_command(task=task, algorithm=algorithm, rounds_to_run=rounds_to_run, seed=seed, base_port=base_port, runtime_device=runtime_device, startup_wait_seconds=startup_wait_seconds, poll_seconds=poll_seconds, output_root=output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True, cwd=_repo_root(), env=env)
    metrics_path = run_dir / 'metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f'Experiment finished but metrics.json is missing: {metrics_path}')
    return run_dir


def build_round_analysis(record: dict[str, Any], config: dict[str, Any], model_state: dict[str, Any]) -> dict[str, Any]:
    algorithm = _algorithm_name(config, record)
    task_type = _task_type(config, record)
    clients = list(record.get('clients') or [])
    payload_kinds = sorted({str(item.get('aggregation_payload_kind', 'unknown')) for item in clients})
    compressors = sorted({str(item.get('compressor', 'none')) for item in clients})
    primary_payload_kind = payload_kinds[0] if len(payload_kinds) == 1 else 'mixed'
    return {
        'round': int(record['round']),
        'task_type': task_type,
        'model_name': str(config.get('model', {}).get('name', 'unknown')),
        'algorithm': algorithm,
        'payload_kind': primary_payload_kind,
        'compressors': compressors,
        'model': {
            'parameters': int(record.get('model_parameters', 0)),
            'bytes': int(record.get('model_bytes', 0)),
            'bytes_human': _format_bytes(int(record.get('model_bytes', 0))),
        },
        'model_state': model_state,
        'parameter': {
            'download_bytes': int(record.get('total_parameter_download_bytes', 0)),
            'upload_bytes': int(record.get('total_parameter_upload_bytes', 0)),
            'total_bytes': int(record.get('total_parameter_bytes', 0)),
            'download_includes': ['protocol-visible download_state payload delivered from server to each client'],
            'download_context_hints': _round_context_hints(config, int(record['round']), algorithm),
            'upload_includes': _parameter_upload_hints(algorithm, primary_payload_kind),
        },
        'transport': {
            'download_bytes': int(record.get('total_transport_download_bytes', 0)),
            'upload_bytes': int(record.get('total_transport_upload_bytes', 0)),
            'total_bytes': int(record.get('total_transport_bytes', 0)),
            'download_overhead_bytes': int(record.get('total_transport_download_overhead_bytes', 0)),
            'upload_overhead_bytes': int(record.get('total_transport_upload_overhead_bytes', 0)),
            'download_includes': _transport_download_hints(),
            'upload_includes': _transport_upload_hints(),
        },
        'protocol_shapes': {
            'parameter_scope': _parameter_scope_summary(algorithm),
            'transport_scope': _transport_scope_summary(),
            'external_scope': _external_scope_summary(),
            'server_to_client_message': {
                'rpc_method': 'GetGlobal',
                'root_keys': ['round', 'state', 'compressed', 'round_context', 'stop'],
                'round_context_keys': _common_round_context_keys(algorithm),
            },
            'client_to_server_message': {
                'rpc_method': 'SubmitUpdate',
                **_submit_update_keys(algorithm),
            },
        },
        'clients': [
            {
                'client_id': str(item.get('client_id')),
                'num_samples': int(item.get('num_samples', 0)),
                'payload_kind': str(item.get('aggregation_payload_kind', 'unknown')),
                'compressor': str(item.get('compressor', 'none')),
                'parameter_download_bytes': int(item.get('parameter_download_bytes', 0)),
                'parameter_upload_bytes': int(item.get('parameter_upload_bytes', 0)),
                'transport_download_bytes': int(item.get('transport_download_bytes', 0)),
                'transport_upload_bytes': int(item.get('transport_upload_bytes', 0)),
            }
            for item in clients
        ],
        'notes': [
            'parameter_* counts algorithm-visible payload bytes, not actual TCP/IP bytes',
            'transport_* counts serialized framework message bytes, including envelopes and metadata',
        ],
    }


def build_report(*, run_dir: Path, rounds: list[int] | None = None, first_n: int | None = None, config_path: Path | None = None, monitor_summary_path: Path | None = None, pcap_path: Path | None = None, run_log_path: Path | None = None, tshark_bin: str = 'tshark', port: int | None = None, local_ips: set[str] | None = None) -> dict[str, Any]:
    history = _load_round_history(run_dir)
    selected_rounds = select_round_indices(history, rounds, first_n)
    selected_records = [item for item in history if int(item['round']) in set(selected_rounds)]
    resolved_config_path = config_path or _discover_config_path(run_dir)
    config = _load_config_artifact(resolved_config_path)
    model_state = _safe_model_state_key_examples(config)
    external = _build_external_capture_report(run_dir=run_dir, selected_rounds=selected_rounds, monitor_summary_path=monitor_summary_path or _discover_monitor_summary(run_dir), pcap_path=pcap_path or _discover_pcap(run_dir), run_log_path=run_log_path or _discover_run_log(run_dir), tshark_bin=tshark_bin, port=port, local_ips=local_ips)
    external_per_round = external.get('per_round') or {}
    rounds_payload = []
    for record in selected_records:
        round_index = int(record['round'])
        analysis = build_round_analysis(record, config, model_state)
        if round_index in external_per_round:
            analysis['external_capture'] = external_per_round[round_index]
            analysis['external_capture']['notes'] = ['frame bytes include lower-layer protocol overhead visible to the capture point', 'tcp payload bytes are closer to application body but still reflect packetization and retransmission effects']
        else:
            analysis['external_capture'] = None
        rounds_payload.append(analysis)
    return {
        'run_dir': str(run_dir),
        'config_path': None if resolved_config_path is None else str(resolved_config_path),
        'selected_rounds': selected_rounds,
        'task_clients': list(config.get('data', {}).get('clients') or []),
        'external_capture': external,
        'rounds': rounds_payload,
    }


def render_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"实验目录: {report['run_dir']}")
    lines.append(f"分析轮次: {report['selected_rounds']}")
    if report.get('task_clients'):
        lines.append(f"参与客户端: {', '.join(report['task_clients'])}")
    external = report.get('external_capture') or {}
    if external.get('overall') is not None:
        overall = external['overall']
        lines.append('外置抓包总体统计: ' f"frame_total={overall.get('total_bytes', 0)} ({_format_bytes(int(overall.get('total_bytes', 0) or 0))}), " f"tcp_payload_total={overall.get('total_tcp_payload_bytes', 0)} ({_format_bytes(int(overall.get('total_tcp_payload_bytes', 0) or 0))})")
    for note in external.get('notes') or []:
        lines.append(f'外置说明: {note}')
    for round_payload in report.get('rounds') or []:
        lines.append('')
        lines.append(f"第 {round_payload['round']} 轮 task={round_payload['task_type']} model={round_payload['model_name']} algorithm={round_payload['algorithm']} ({ALGORITHM_LABELS.get(round_payload['algorithm'], round_payload['algorithm'])}) payload={round_payload['payload_kind']}")
        model_state = round_payload.get('model_state') or {}
        if model_state.get('build_error'):
            lines.append(f"  模型 state_dict: 无法构建模型，原因={model_state['build_error']}")
        elif model_state:
            lines.append('  模型 state_dict: ' f"class={model_state.get('model_class')} keys={model_state.get('state_dict_key_count')} " f"trainable={model_state.get('trainable_key_count')} buffer={model_state.get('buffer_key_count')}")
            for item in model_state.get('trainable_key_examples') or []:
                lines.append(f"    trainable key: {item['key']} shape={item['shape']} dtype={item['dtype']} bytes={item['bytes']}")
            for item in model_state.get('buffer_key_examples') or []:
                lines.append(f"    buffer key: {item['key']} shape={item['shape']} dtype={item['dtype']} bytes={item['bytes']}")
        lines.append('  Parameter 口径: ' f"download={round_payload['parameter']['download_bytes']} ({_format_bytes(round_payload['parameter']['download_bytes'])}), " f"upload={round_payload['parameter']['upload_bytes']} ({_format_bytes(round_payload['parameter']['upload_bytes'])}), " f"total={round_payload['parameter']['total_bytes']} ({_format_bytes(round_payload['parameter']['total_bytes'])})")
        parameter_scope = round_payload['protocol_shapes']['parameter_scope']
        lines.append(f"    统计公式: {parameter_scope['download_formula']}")
        for item in parameter_scope.get('download_counts') or []:
            lines.append(f'    具体统计下载内容: {item}')
        for item in parameter_scope.get('upload_counts') or []:
            lines.append(f'    具体统计上传内容: {item}')
        for item in parameter_scope.get('upload_tensor_container') or []:
            lines.append(f'    对应字段/容器: {item}')
        for item in round_payload['parameter']['download_includes']:
            lines.append(f'    下载说明: {item}')
        for item in round_payload['parameter']['download_context_hints']:
            lines.append(f'    round_context 说明: {item}')
        for item in round_payload['parameter']['upload_includes']:
            lines.append(f'    上传说明: {item}')
        lines.append('  Transport 口径: ' f"download={round_payload['transport']['download_bytes']} ({_format_bytes(round_payload['transport']['download_bytes'])}), " f"upload={round_payload['transport']['upload_bytes']} ({_format_bytes(round_payload['transport']['upload_bytes'])}), " f"total={round_payload['transport']['total_bytes']} ({_format_bytes(round_payload['transport']['total_bytes'])}), " f"download_overhead={round_payload['transport']['download_overhead_bytes']} ({_format_bytes(round_payload['transport']['download_overhead_bytes'])}), " f"upload_overhead={round_payload['transport']['upload_overhead_bytes']} ({_format_bytes(round_payload['transport']['upload_overhead_bytes'])})")
        transport_scope = round_payload['protocol_shapes']['transport_scope']
        lines.append(f"    GetGlobal 根字段: {', '.join(transport_scope['download_rpc_envelope_keys'])}")
        lines.append(f"    SubmitUpdate 根字段: {', '.join(transport_scope['upload_rpc_envelope_keys'])}")
        for item in transport_scope.get('extra_counted_bytes') or []:
            lines.append(f'    额外统计内容: {item}')
        server_to_client = round_payload['protocol_shapes']['server_to_client_message']
        lines.append(f"    round_context keys: {', '.join(server_to_client['round_context_keys'])}")
        client_to_server = round_payload['protocol_shapes']['client_to_server_message']
        lines.append(f"    ClientResult 关键字段: {', '.join(client_to_server['result_primary_keys'])}")
        if client_to_server.get('ega_payload_keys'):
            lines.append(f"    EGA payload 关键字段: {', '.join(client_to_server['ega_payload_keys'])}")
        for item in round_payload['transport']['download_includes']:
            lines.append(f'    transport 下载说明: {item}')
        for item in round_payload['transport']['upload_includes']:
            lines.append(f'    transport 上传说明: {item}')
        external_capture = round_payload.get('external_capture')
        if external_capture is None:
            lines.append('  外置抓包: 当前轮不可用')
        else:
            lines.append('  外置抓包(按窗口近似切片): ' f"frame_total={external_capture['total_bytes']} ({_format_bytes(external_capture['total_bytes'])}), " f"tcp_payload_total={external_capture['total_tcp_payload_bytes']} ({_format_bytes(external_capture['total_tcp_payload_bytes'])}), " f"sent_packets={external_capture['sent_packets']}, received_packets={external_capture['received_packets']}")
            for item in external_capture.get('notes') or []:
                lines.append(f'    外置说明: {item}')
        for item in round_payload['protocol_shapes']['external_scope']['summary_json_keys']:
            lines.append(f'    grpc_port_traffic.summary.json key: {item}')
        for item in round_payload['protocol_shapes']['external_scope']['pcap_note']:
            lines.append(f'    抓包口径说明: {item}')
        for item in round_payload.get('notes') or []:
            lines.append(f'  备注: {item}')
        lines.append('  客户端分解:')
        for client in round_payload.get('clients') or []:
            lines.append(f"    {client['client_id']}: payload={client['payload_kind']} compressor={client['compressor']} parameter(down/up)={client['parameter_download_bytes']}/{client['parameter_upload_bytes']} transport(down/up)={client['transport_download_bytes']}/{client['transport_upload_bytes']}")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='Explain parameter/transport/external communication for selected federated rounds')
    parser.add_argument('run_dir', nargs='?', type=Path, help='Experiment directory containing metrics.json')
    parser.add_argument('--run-experiment', action='store_true', help='Launch a fresh multi-sync grpc experiment first, then analyze it')
    parser.add_argument('--task', choices=sorted(TASK_CONFIG_PATHS), default='rare', help='Task used when --run-experiment is enabled')
    parser.add_argument('--algorithm', choices=['fedavg', 'topk', 'ega'], default='ega', help='Algorithm used when --run-experiment is enabled')
    parser.add_argument('--rounds-to-run', type=int, default=1, help='How many grpc_sync rounds to execute when --run-experiment is enabled')
    parser.add_argument('--seed', type=int, default=2026, help='Seed used when --run-experiment is enabled')
    parser.add_argument('--base-port', type=int, default=58000, help='Base port passed to scripts/run_suite.sh for fresh experiments')
    parser.add_argument('--runtime-device', default='cuda:0', help='runtime.device override for fresh experiments')
    parser.add_argument('--startup-wait-seconds', type=int, default=5, help='Wait time before spawning grpc clients in fresh experiments')
    parser.add_argument('--poll-seconds', type=float, default=1.0, help='grpc.poll_seconds override for fresh experiments')
    parser.add_argument('--output-root', type=Path, default=None, help='Fresh experiment root; defaults to outputs/communication_analysis/seed_xxx_task_algo_rx')
    parser.add_argument('--round', dest='rounds', type=int, action='append', default=None, help='Specific round index to analyze; repeatable')
    parser.add_argument('--first-n', type=int, default=None, help='Analyze only the first N rounds when --round is not provided')
    parser.add_argument('--config', type=Path, default=None, help='Optional config artifact path; defaults to run_dir/config.yaml or config.json')
    parser.add_argument('--monitor-dir', type=Path, default=None, help='Directory containing grpc_port_traffic.summary.json and grpc_port_traffic.pcap')
    parser.add_argument('--monitor-summary', type=Path, default=None, help='Optional grpc_port_traffic.summary.json path')
    parser.add_argument('--pcap', type=Path, default=None, help='Optional grpc_port_traffic.pcap path')
    parser.add_argument('--run-log', type=Path, default=None, help='Optional run.log path used for approximate external per-round slicing')
    parser.add_argument('--tshark-bin', default='tshark', help='tshark executable used to parse pcap when available')
    parser.add_argument('--port', type=int, default=None, help='Override monitored server port for pcap parsing')
    parser.add_argument('--local-ip', action='append', default=None, help='Additional local/server IPs used to classify packet direction; repeatable')
    parser.add_argument('--json', action='store_true', help='Print the machine-readable JSON report instead of the human-readable explanation')
    parser.add_argument('--output', type=Path, default=None, help='Optional output file path receiving the rendered report')
    args = parser.parse_args()

    run_dir = None if args.run_dir is None else args.run_dir.expanduser().resolve()
    if args.run_experiment:
        output_root = args.output_root.expanduser().resolve() if args.output_root is not None else (_repo_root() / 'outputs' / 'communication_analysis' / f'seed_{args.seed}_{args.task}_{args.algorithm}_r{args.rounds_to_run}')
        run_dir = run_experiment_and_resolve_run_dir(task=str(args.task), algorithm=str(args.algorithm), rounds_to_run=int(args.rounds_to_run), seed=int(args.seed), base_port=int(args.base_port), runtime_device=str(args.runtime_device), startup_wait_seconds=int(args.startup_wait_seconds), poll_seconds=float(args.poll_seconds), output_root=output_root)
        if args.first_n is None and not args.rounds:
            args.first_n = int(args.rounds_to_run)
    if run_dir is None:
        raise ValueError('run_dir is required unless --run-experiment is enabled')

    monitor_summary_path = args.monitor_summary
    pcap_path = args.pcap
    if args.monitor_dir is not None:
        if monitor_summary_path is None:
            monitor_summary_path = args.monitor_dir / 'grpc_port_traffic.summary.json'
        if pcap_path is None:
            pcap_path = args.monitor_dir / 'grpc_port_traffic.pcap'
    report = build_report(run_dir=run_dir, rounds=None if not args.rounds else list(args.rounds), first_n=args.first_n, config_path=None if args.config is None else args.config.expanduser().resolve(), monitor_summary_path=None if monitor_summary_path is None else monitor_summary_path.expanduser().resolve(), pcap_path=None if pcap_path is None else pcap_path.expanduser().resolve(), run_log_path=None if args.run_log is None else args.run_log.expanduser().resolve(), tshark_bin=str(args.tshark_bin), port=args.port, local_ips=None if not args.local_ip else set(args.local_ip))
    rendered = json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_report(report)
    if args.output is not None:
        args.output.expanduser().resolve().write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
