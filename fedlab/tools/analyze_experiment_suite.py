#!/usr/bin/env python3
"""Summarize and plot selected experiment-suite results.
The tool scans one or more suite directories, selects runs by algorithm-name
matching, then exports tabular summaries plus validation/test metric
visualizations.
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
RUN_NAME_MARKERS = (
    '_single_sync_uupdate_dmodel_',
    '_single_async_uupdate_dmodel_',
    '_grpc_sync_uupdate_dmodel_',
    '_grpc_async_uupdate_dmodel_',
    '_single_sync_',
    '_single_async_',
    '_grpc_sync_',
    '_grpc_async_',
)
DEFAULT_ALGORITHMS = ('centralized', 'fedavg', 'topk', 'ega')
SUPPORTED_METRICS = ('mse', 'mae', 'mape', 'accuracy')
BAR_COLORS = {
    'centralized': '#4c78a8',
    'topk': '#f58518',
    'ega': '#54a24b',
}
LEGACY_MSE_OUTPUTS = (
    'val_mse_vs_round.png',
    'val_mse_vs_cumulative_upload.png',
    'test_mse_bar.png',
    'test_mse_vs_upload_bubble.png',
)

METRIC_TITLES = {
    'mse': 'MSE',
    'mae': 'MAE',
    'mape': 'MAPE',
    'accuracy': 'Accuracy',
}
METRIC_OPTIMIZATION_OBJECTIVES = {
    'accuracy': 'max',
}

def _metric_optimization_objective(metric: str) -> str:
    return METRIC_OPTIMIZATION_OBJECTIVES.get(metric, 'min')

def _best_so_far_values(values: list[float], metric: str) -> list[float]:
    if not values:
        return []
    objective = _metric_optimization_objective(metric)
    best_values: list[float] = []
    best = values[0]
    for value in values:
        if objective == 'max':
            best = max(best, value)
        else:
            best = min(best, value)
        best_values.append(best)
    return best_values
@dataclass
class RunRecord:
    label: str
    run_name: str
    run_dir: Path
    algorithm: str
    val_series: dict[str, list[tuple[int, float, int]]]
    test_metrics: dict[str, float | None]
    total_upload_bytes: int
    attack_present: bool
    attack_primary_metric_name: str | None
    attack_primary_metric_direction: str | None
    attack_overall_avg_primary_metric_value: float | None
    attack_success_rate: float | None
    attack_evaluations: int | None
    @property
    def val_rounds(self) -> list[int]:
        return [round_id for round_id, _value, _upload in self.val_series.get('mse', [])]
    @property
    def val_mse(self) -> list[float]:
        return [value for _round_id, value, _upload in self.val_series.get('mse', [])]
    @property
    def val_cumulative_upload_bytes(self) -> list[int]:
        return [upload for _round_id, _value, upload in self.val_series.get('mse', [])]
    @property
    def test_mse(self) -> float | None:
        return self.test_metrics.get('mse')
@dataclass
class AggregatedCurve:
    label: str
    rounds: list[int]
    round_mean: list[float]
    round_std: list[float]
    upload_mean: list[float]
    upload_std: list[float]
def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)
def resolve_suite_dir(root_dir: Path, loss: str | None) -> Path:
    if loss is None:
        return root_dir
    for name in (loss, f'noattack_{loss}', f'attack_{loss}'):
        candidate = root_dir / name
        if candidate.is_dir():
            return candidate
    return root_dir
def default_output_dir(root_dir: Path, loss: str | None) -> Path:
    suffix = loss or 'all'
    return root_dir.parent / f'{root_dir.name}_analysis_{suffix}'
def default_multi_output_dir(root_dirs: list[Path], loss: str | None) -> Path:
    suffix = loss or 'all'
    common_parent = root_dirs[0].parent
    return common_parent / f'multiseed_analysis_{suffix}'
def normalize_algorithm_tokens(algorithms: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(token.strip().lower() for token in algorithms if token.strip())
def _normalize_metric_tokens(metrics: tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
    if metrics is None:
        return None
    normalized: list[str] = []
    for metric in metrics:
        value = str(metric).strip().lower()
        if value and value in SUPPORTED_METRICS and value not in normalized:
            normalized.append(value)
    return tuple(normalized)
def is_selected_run_dir(path: Path, algorithms: tuple[str, ...], include_old: bool) -> bool:
    if not path.is_dir():
        return False
    if not include_old and path.name.endswith('_old'):
        return False
    if not (path / 'summary.json').exists() or not (path / 'metrics.json').exists():
        return False
    lower_name = path.name.lower()
    return any(token in lower_name for token in algorithms)
def canonical_label(run_name: str) -> str:
    is_old = run_name.endswith('_old')
    base_name = run_name[:-4] if is_old else run_name
    if base_name.startswith('centralized_'):
        label = 'centralized'
    elif base_name.startswith('fedavg_'):
        label = 'fedavg'
    elif base_name.startswith('topk_'):
        label = 'topk'
    elif base_name.startswith('ega_'):
        label = base_name
        for marker in RUN_NAME_MARKERS:
            if marker in base_name:
                label = base_name.split(marker, 1)[0]
                break
    else:
        label = base_name
    return f'{label}_old' if is_old else label
def _resolve_validation_metric(record: dict[str, Any], metric: str) -> float | None:
    for container_key in ('val_metrics', 'protocol_val_metrics', 'active_val_metrics'):
        container = record.get(container_key)
        if isinstance(container, dict) and container.get(metric) is not None:
            return float(container[metric])
    for key in (f'active_val_{metric}', f'protocol_val_{metric}', f'val_{metric}'):
        if key in record and record[key] is not None:
            return float(record[key])
    if metric == 'accuracy' and record.get('primary_metric_name') == 'accuracy' and record.get('primary_metric_value') is not None:
        return float(record['primary_metric_value'])
    return None
def _resolve_round_upload_bytes(record: dict[str, Any]) -> int:
    for key in ('total_parameter_upload_bytes', 'total_upload_bytes'):
        if key in record and record[key] is not None:
            return int(record[key])
    return 0
def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)
def _unique_or_mixed(values: list[str | None]) -> str | None:
    cleaned = [value for value in values if value]
    if not cleaned:
        return None
    unique: list[str] = []
    for value in cleaned:
        if value not in unique:
            unique.append(value)
    if len(unique) == 1:
        return unique[0]
    return 'mixed'
def _label_family(label: str) -> str:
    if label.startswith('centralized'):
        return 'centralized'
    if label.startswith('fedavg'):
        return 'fedavg'
    if label.startswith('topk'):
        return 'topk'
    if label.startswith('ega'):
        return 'ega'
    return 'other'
def _bar_color(label: str) -> str:
    return BAR_COLORS.get(_label_family(label), '#9c755f')
def _bubble_size_from_ratio(ratio: float | None) -> float:
    if ratio is None or ratio <= 0.0:
        return 180.0
    return 180.0 + 120.0 * ratio
def _relative_percent(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0.0:
        return None
    return (value - baseline) / baseline * 100.0
def _metric_key(metric: str) -> str:
    return f'test_{metric}'
def _metric_relative_key(metric: str) -> str:
    return f'{metric}_relative_percent_vs_centralized'
def _metric_mean_key(metric: str) -> str:
    return f'test_{metric}_mean'
def _metric_std_key(metric: str) -> str:
    return f'test_{metric}_std'
def _metric_relative_mean_key(metric: str) -> str:
    return f'{metric}_relative_percent_vs_centralized_mean'
def _metric_relative_std_key(metric: str) -> str:
    return f'{metric}_relative_percent_vs_centralized_std'
def _resolve_attack_info(summary: dict[str, Any]) -> dict[str, Any]:
    attack_summary = summary.get('attack_summary') or {}
    target_type = summary.get('attack_target_type')
    primary_metric_name = (
        summary.get('attack_primary_metric_name')
        or summary.get('attack_primary_metric')
        or attack_summary.get('primary_metric_name')
        or attack_summary.get('primary_metric')
    )
    primary_metric_direction = (
        summary.get('attack_primary_metric_direction')
        or attack_summary.get('primary_metric_direction')
    )
    avg_primary_value = summary.get('attack_overall_avg_primary_metric_value')
    if avg_primary_value is None:
        avg_primary_value = summary.get('attack_overall_avg_primary_metric')
    if avg_primary_value is None and primary_metric_name:
        avg_primary_value = attack_summary.get(f'overall_avg_{primary_metric_name}')
    success_rate = summary.get('attack_success_rate')
    if success_rate is None:
        success_rate = attack_summary.get('overall_success_rate')
    evaluations = summary.get('attack_evaluations')
    present = bool(target_type) or bool((evaluations or 0) > 0)
    return {
        'attack_present': present,
        'attack_primary_metric_name': str(primary_metric_name) if primary_metric_name is not None and present else None,
        'attack_primary_metric_direction': str(primary_metric_direction) if primary_metric_direction is not None and present else None,
        'attack_overall_avg_primary_metric_value': float(avg_primary_value) if avg_primary_value is not None and present else None,
        'attack_success_rate': float(success_rate) if success_rate is not None and present else None,
        'attack_evaluations': int(evaluations) if evaluations is not None and present else None,
    }
def load_run(run_dir: Path) -> RunRecord:
    metrics = load_json(run_dir / 'metrics.json')
    summary = load_json(run_dir / 'summary.json')
    val_series: dict[str, list[tuple[int, float, int]]] = {metric: [] for metric in SUPPORTED_METRICS}
    if isinstance(metrics, dict):
        history = metrics.get('history') or []
        if not isinstance(history, list):
            raise TypeError(f'Unsupported centralized metrics format in {run_dir}')
        for idx, item in enumerate(history):
            round_id = int(item.get('round', idx))
            for metric in SUPPORTED_METRICS:
                value = _resolve_validation_metric(item, metric)
                if value is not None:
                    val_series[metric].append((round_id, value, 0))
        algorithm = str(summary.get('algorithm') or 'centralized')
    elif isinstance(metrics, list):
        running = 0
        algorithm = str(summary.get('algorithm') or (metrics[0].get('algorithm') if metrics else 'unknown') or 'unknown')
        for idx, item in enumerate(metrics):
            running += _resolve_round_upload_bytes(item)
            round_id = int(item.get('round', idx))
            for metric in SUPPORTED_METRICS:
                value = _resolve_validation_metric(item, metric)
                if value is not None:
                    val_series[metric].append((round_id, value, running))
    else:
        raise TypeError(f'Unsupported metrics format in {run_dir}: {type(metrics)!r}')
    test_blob = summary.get('test') or summary.get('protocol_test') or {}
    test_metrics = {metric: (float(test_blob[metric]) if metric in test_blob and test_blob[metric] is not None else None) for metric in SUPPORTED_METRICS}
    total_upload_bytes = int(summary.get('total_parameter_upload_bytes', summary.get('total_upload_bytes', 0)) or 0)
    attack_info = _resolve_attack_info(summary)
    return RunRecord(
        label=canonical_label(run_dir.name),
        run_name=run_dir.name,
        run_dir=run_dir,
        algorithm=algorithm,
        val_series=val_series,
        test_metrics=test_metrics,
        total_upload_bytes=total_upload_bytes,
        attack_present=bool(attack_info['attack_present']),
        attack_primary_metric_name=attack_info['attack_primary_metric_name'],
        attack_primary_metric_direction=attack_info['attack_primary_metric_direction'],
        attack_overall_avg_primary_metric_value=attack_info['attack_overall_avg_primary_metric_value'],
        attack_success_rate=attack_info['attack_success_rate'],
        attack_evaluations=attack_info['attack_evaluations'],
    )
def discover_runs(root_dir: Path, loss: str | None, algorithms: tuple[str, ...], include_old: bool) -> list[RunRecord]:
    suite_dir = resolve_suite_dir(root_dir, loss)
    run_dirs = sorted(path for path in suite_dir.iterdir() if is_selected_run_dir(path, algorithms, include_old))
    records = [load_run(path) for path in run_dirs]
    order = {'centralized': 0, 'fedavg': 1, 'topk': 2}
    return sorted(records, key=lambda record: (order.get(record.label.replace('_old', ''), 3), record.label))
def infer_metrics_from_records(records: list[RunRecord]) -> tuple[str, ...]:
    metrics: list[str] = []
    for metric in SUPPORTED_METRICS:
        if any(record.test_metrics.get(metric) is not None or record.val_series.get(metric) for record in records):
            metrics.append(metric)
    return tuple(metrics)
def _resolve_metrics(records: list[RunRecord], metrics: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    normalized = _normalize_metric_tokens(metrics)
    if normalized:
        return normalized
    inferred = infer_metrics_from_records(records)
    return inferred or ('mse',)
def build_rows(records: list[RunRecord], metrics: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    selected_metrics = _resolve_metrics(records, metrics)
    fedavg = next((record for record in records if record.label == 'fedavg'), None)
    centralized = next((record for record in records if record.label == 'centralized'), None)
    if fedavg is None:
        raise ValueError('FedAvg run is required to compute compression ratios')
    if centralized is None:
        raise ValueError('Centralized run is required to compute performance loss ratios')
    rows = []
    for record in records:
        compression_ratio = None
        if record.total_upload_bytes > 0:
            compression_ratio = fedavg.total_upload_bytes / record.total_upload_bytes
        row: dict[str, Any] = {
            'label': record.label,
            'run_name': record.run_name,
            'total_upload_bytes': record.total_upload_bytes,
            'fedavg_upload_compression_ratio': compression_ratio,
            'selected_metrics': ','.join(selected_metrics),
            'attack_primary_metric_name': record.attack_primary_metric_name if record.attack_present else None,
            'attack_primary_metric_direction': record.attack_primary_metric_direction if record.attack_present else None,
            'attack_overall_avg_primary_metric_value': record.attack_overall_avg_primary_metric_value if record.attack_present else None,
            'attack_success_rate': record.attack_success_rate if record.attack_present else None,
            'attack_evaluations': record.attack_evaluations if record.attack_present else None,
        }
        for metric in selected_metrics:
            value = record.test_metrics.get(metric)
            baseline = centralized.test_metrics.get(metric)
            row[_metric_key(metric)] = value
            row[_metric_relative_key(metric)] = _relative_percent(value, baseline)
        if 'mse' in selected_metrics:
            row['test_mse'] = row.get('test_mse') if 'test_mse' in row else row[_metric_key('mse')]
            row['mse_loss_ratio_percent'] = row[_metric_relative_key('mse')]
        else:
            row['test_mse'] = None
            row['mse_loss_ratio_percent'] = None
        rows.append(row)
    return rows
def aggregate_rows(records_by_seed: list[list[RunRecord]], metrics: tuple[str, ...] | list[str] | None = None) -> list[dict[str, Any]]:
    selected_metrics = _resolve_metrics([record for records in records_by_seed for record in records], metrics)
    rows_by_seed = [build_rows(records, selected_metrics) for records in records_by_seed]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in rows_by_seed:
        for row in rows:
            grouped[str(row['label'])].append(row)
    seed_order: list[str] = []
    for rows in rows_by_seed:
        for row in rows:
            label = str(row['label'])
            if label not in seed_order:
                seed_order.append(label)
    aggregated_rows = []
    for label in seed_order:
        entries = grouped[label]
        upload_values = [float(entry['total_upload_bytes']) for entry in entries if entry['total_upload_bytes'] is not None]
        ratio_values = [float(entry['fedavg_upload_compression_ratio']) for entry in entries if entry['fedavg_upload_compression_ratio'] is not None]
        attack_metric_values = [float(entry['attack_overall_avg_primary_metric_value']) for entry in entries if entry['attack_overall_avg_primary_metric_value'] is not None]
        attack_success_values = [float(entry['attack_success_rate']) for entry in entries if entry['attack_success_rate'] is not None]
        attack_eval_values = [float(entry['attack_evaluations']) for entry in entries if entry['attack_evaluations'] is not None]
        upload_mean, upload_std = _mean_std(upload_values)
        ratio_mean, ratio_std = _mean_std(ratio_values)
        attack_metric_mean, attack_metric_std = _mean_std(attack_metric_values)
        attack_success_mean, attack_success_std = _mean_std(attack_success_values)
        attack_eval_mean, attack_eval_std = _mean_std(attack_eval_values)
        row: dict[str, Any] = {
            'label': label,
            'seed_count': len(entries),
            'total_upload_bytes_mean': upload_mean,
            'total_upload_bytes_std': upload_std,
            'fedavg_upload_compression_ratio_mean': ratio_mean,
            'fedavg_upload_compression_ratio_std': ratio_std,
            'selected_metrics': ','.join(selected_metrics),
            'attack_seed_count': len(attack_metric_values) or len(attack_success_values) or len(attack_eval_values),
            'attack_primary_metric_name': _unique_or_mixed([entry['attack_primary_metric_name'] for entry in entries]),
            'attack_primary_metric_direction': _unique_or_mixed([entry['attack_primary_metric_direction'] for entry in entries]),
            'attack_overall_avg_primary_metric_value_mean': attack_metric_mean,
            'attack_overall_avg_primary_metric_value_std': attack_metric_std,
            'attack_success_rate_mean': attack_success_mean,
            'attack_success_rate_std': attack_success_std,
            'attack_evaluations_mean': attack_eval_mean,
            'attack_evaluations_std': attack_eval_std,
        }
        for metric in selected_metrics:
            test_values = [float(entry[_metric_key(metric)]) for entry in entries if entry.get(_metric_key(metric)) is not None]
            rel_values = [float(entry[_metric_relative_key(metric)]) for entry in entries if entry.get(_metric_relative_key(metric)) is not None]
            test_mean, test_std = _mean_std(test_values)
            rel_mean, rel_std = _mean_std(rel_values)
            row[_metric_mean_key(metric)] = test_mean
            row[_metric_std_key(metric)] = test_std
            row[_metric_relative_mean_key(metric)] = rel_mean
            row[_metric_relative_std_key(metric)] = rel_std
        if 'mse' in selected_metrics:
            row['test_mse_mean'] = row[_metric_mean_key('mse')]
            row['test_mse_std'] = row[_metric_std_key('mse')]
            row['mse_loss_ratio_percent_mean'] = row[_metric_relative_mean_key('mse')]
            row['mse_loss_ratio_percent_std'] = row[_metric_relative_std_key('mse')]
        else:
            row['test_mse_mean'] = None
            row['test_mse_std'] = None
            row['mse_loss_ratio_percent_mean'] = None
            row['mse_loss_ratio_percent_std'] = None
        aggregated_rows.append(row)
    return aggregated_rows
def aggregate_metric_curves(records_by_seed: list[list[RunRecord]], metric: str) -> list[AggregatedCurve]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    label_order: list[str] = []
    for records in records_by_seed:
        for record in records:
            if record.label not in grouped:
                label_order.append(record.label)
            grouped[record.label].append(record)
    curves: list[AggregatedCurve] = []
    for label in label_order:
        round_to_val: dict[int, list[float]] = defaultdict(list)
        round_to_upload: dict[int, list[float]] = defaultdict(list)
        for record in grouped[label]:
            for round_id, value, upload in record.val_series.get(metric, []):
                round_to_val[round_id].append(value)
                round_to_upload[round_id].append(float(upload))
        rounds = sorted(round_to_val.keys())
        round_mean: list[float] = []
        round_std: list[float] = []
        upload_mean: list[float] = []
        upload_std: list[float] = []
        kept_rounds: list[int] = []
        for round_id in rounds:
            value_mean, value_std = _mean_std(round_to_val[round_id])
            upload_mean_value, upload_std_value = _mean_std(round_to_upload[round_id])
            if value_mean is None or value_std is None or upload_mean_value is None or upload_std_value is None:
                continue
            kept_rounds.append(round_id)
            round_mean.append(value_mean)
            round_std.append(value_std)
            upload_mean.append(upload_mean_value)
            upload_std.append(upload_std_value)
        curves.append(AggregatedCurve(
            label=label,
            rounds=kept_rounds,
            round_mean=round_mean,
            round_std=round_std,
            upload_mean=upload_mean,
            upload_std=upload_std,
        ))
    return curves
def aggregate_best_metric_curves(records_by_seed: list[list[RunRecord]], metric: str) -> list[AggregatedCurve]:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    label_order: list[str] = []
    for records in records_by_seed:
        for record in records:
            if record.label not in grouped:
                label_order.append(record.label)
            grouped[record.label].append(record)
    curves: list[AggregatedCurve] = []
    for label in label_order:
        round_to_val: dict[int, list[float]] = defaultdict(list)
        round_to_upload: dict[int, list[float]] = defaultdict(list)
        for record in grouped[label]:
            rounds, values, uploads = _validation_series(record, metric)
            best_values = _best_so_far_values(values, metric)
            for round_id, value, upload in zip(rounds, best_values, uploads):
                round_to_val[round_id].append(value)
                round_to_upload[round_id].append(float(upload))
        rounds = sorted(round_to_val.keys())
        round_mean: list[float] = []
        round_std: list[float] = []
        upload_mean: list[float] = []
        upload_std: list[float] = []
        kept_rounds: list[int] = []
        for round_id in rounds:
            value_mean, value_std = _mean_std(round_to_val[round_id])
            upload_mean_value, upload_std_value = _mean_std(round_to_upload[round_id])
            if value_mean is None or value_std is None or upload_mean_value is None or upload_std_value is None:
                continue
            kept_rounds.append(round_id)
            round_mean.append(value_mean)
            round_std.append(value_std)
            upload_mean.append(upload_mean_value)
            upload_std.append(upload_std_value)
        curves.append(AggregatedCurve(
            label=label,
            rounds=kept_rounds,
            round_mean=round_mean,
            round_std=round_std,
            upload_mean=upload_mean,
            upload_std=upload_std,
        ))
    return curves

def aggregate_curves(records_by_seed: list[list[RunRecord]]) -> list[AggregatedCurve]:
    return aggregate_metric_curves(records_by_seed, 'mse')
def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'

def format_latex_float(value: float | None, digits: int = 6, *, signed: bool = False, suffix: str = '') -> str:
    if value is None:
        return 'n/a'
    spec = '+' if signed else ''
    return f'${value:{spec}.{digits}f}{suffix}$'

def format_latex_pm(mean_value: float | None, std_value: float | None, digits: int = 6, *, suffix: str = '') -> str:
    if mean_value is None:
        return 'n/a'
    if std_value is None:
        return format_latex_float(mean_value, digits=digits, suffix=suffix)
    return f'${mean_value:.{digits}f} \\pm {std_value:.{digits}f}{suffix}$'
def format_bytes(value: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    current = float(value)
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f'{current:.2f}{unit}'
        current /= 1024.0
    return f'{current:.2f}TB'
def _metrics_from_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    metrics: list[str] = []
    for metric in SUPPORTED_METRICS:
        if any(_metric_key(metric) in row or _metric_mean_key(metric) in row for row in rows):
            metrics.append(metric)
    return tuple(metrics)
def write_summary_csv(rows: list[dict[str, Any]], output_path: Path, metrics: tuple[str, ...] | list[str] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics = _normalize_metric_tokens(metrics) or _metrics_from_rows(rows)
    fieldnames = [
        'label',
        'run_name',
        'total_upload_bytes',
        'fedavg_upload_compression_ratio',
    ]
    for metric in selected_metrics:
        fieldnames.extend([_metric_key(metric), _metric_relative_key(metric)])
    fieldnames.extend([
        'attack_primary_metric_name',
        'attack_primary_metric_direction',
        'attack_overall_avg_primary_metric_value',
        'attack_success_rate',
        'attack_evaluations',
    ])
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path
def write_aggregated_summary_csv(rows: list[dict[str, Any]], output_path: Path, metrics: tuple[str, ...] | list[str] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics = _normalize_metric_tokens(metrics) or _metrics_from_rows(rows)
    fieldnames = [
        'label',
        'seed_count',
        'total_upload_bytes_mean',
        'total_upload_bytes_std',
        'fedavg_upload_compression_ratio_mean',
        'fedavg_upload_compression_ratio_std',
    ]
    for metric in selected_metrics:
        fieldnames.extend([
            _metric_mean_key(metric),
            _metric_std_key(metric),
            _metric_relative_mean_key(metric),
            _metric_relative_std_key(metric),
        ])
    fieldnames.extend([
        'attack_seed_count',
        'attack_primary_metric_name',
        'attack_primary_metric_direction',
        'attack_overall_avg_primary_metric_value_mean',
        'attack_overall_avg_primary_metric_value_std',
        'attack_success_rate_mean',
        'attack_success_rate_std',
        'attack_evaluations_mean',
        'attack_evaluations_std',
    ])
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return output_path
def write_summary_markdown(rows: list[dict[str, Any]], output_path: Path, metrics: tuple[str, ...] | list[str] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics = _normalize_metric_tokens(metrics) or _metrics_from_rows(rows)
    headers = ['Label', 'Upload', 'FedAvg/F']
    for metric in selected_metrics:
        title = METRIC_TITLES[metric]
        headers.extend([f'Test ${title}$', f'$\\Delta$ {title} vs Centralized $(\\%)$'])
    headers.extend(['Attack Metric', 'Attack Avg', 'Attack Success', 'Attack Evals'])
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] + ['---:' for _ in headers[1:]]) + ' |',
    ]
    for row in rows:
        values = [
            str(row['label']),
            f"{row['total_upload_bytes']} ({format_bytes(int(row['total_upload_bytes']))})",
            format_latex_float(row['fedavg_upload_compression_ratio']),
        ]
        for metric in selected_metrics:
            values.extend([
                format_latex_float(row.get(_metric_key(metric))),
                format_latex_float(row.get(_metric_relative_key(metric)), signed=True, suffix='\\%'),
            ])
        values.extend([
            row['attack_primary_metric_name'] or 'n/a',
            format_latex_float(row['attack_overall_avg_primary_metric_value']),
            format_latex_float(row['attack_success_rate']),
            format_latex_float(float(row['attack_evaluations'])) if row['attack_evaluations'] is not None else 'n/a',
        ])
        lines.append('| ' + ' | '.join(values) + ' |')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path
def write_aggregated_summary_markdown(rows: list[dict[str, Any]], output_path: Path, metrics: tuple[str, ...] | list[str] | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_metrics = _normalize_metric_tokens(metrics) or _metrics_from_rows(rows)
    headers = ['Label', 'Seeds', 'Upload', 'FedAvg/F']
    for metric in selected_metrics:
        title = METRIC_TITLES[metric]
        headers.extend([f'Test ${title}$', f'$\\Delta$ {title} vs Centralized $(\\%)$'])
    headers.extend(['Attack Metric', 'Attack Seeds', 'Attack Avg', 'Attack Success', 'Attack Evals'])
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] + ['---:' for _ in headers[1:]]) + ' |',
    ]
    for row in rows:
        upload_mean = int(round(row['total_upload_bytes_mean'])) if row['total_upload_bytes_mean'] is not None else None
        values = [
            str(row['label']),
            str(row['seed_count']),
            (f"{upload_mean} ({format_bytes(upload_mean)}), $\\pm {format_float(row['total_upload_bytes_std'])}$" if upload_mean is not None else 'n/a'),
            format_latex_pm(row['fedavg_upload_compression_ratio_mean'], row['fedavg_upload_compression_ratio_std']),
        ]
        for metric in selected_metrics:
            values.extend([
                format_latex_pm(row.get(_metric_mean_key(metric)), row.get(_metric_std_key(metric))),
                format_latex_pm(row.get(_metric_relative_mean_key(metric)), row.get(_metric_relative_std_key(metric)), suffix='\\%'),
            ])
        values.extend([
            row['attack_primary_metric_name'] or 'n/a',
            str(row['attack_seed_count']),
            format_latex_pm(row['attack_overall_avg_primary_metric_value_mean'], row['attack_overall_avg_primary_metric_value_std']),
            format_latex_pm(row['attack_success_rate_mean'], row['attack_success_rate_std']),
            format_latex_pm(row['attack_evaluations_mean'], row['attack_evaluations_std']),
        ])
        lines.append('| ' + ' | '.join(values) + ' |')
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path
def _validation_series(record: RunRecord, metric: str) -> tuple[list[int], list[float], list[int]]:
    series = record.val_series.get(metric, [])
    return (
        [round_id for round_id, _value, _upload in series],
        [value for _round_id, value, _upload in series],
        [upload for _round_id, _value, upload in series],
    )
def plot_validation_metric_vs_round(records: list[RunRecord], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    for record in records:
        rounds, values, _uploads = _validation_series(record, metric)
        if not rounds or not values:
            continue
        plt.plot(rounds, values, linewidth=1.8, label=record.label)
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, f'No runs with validation {METRIC_TITLES[metric]}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Round')
    plt.ylabel(f'Validation {METRIC_TITLES[metric]}')
    plt.title(f'Validation {METRIC_TITLES[metric]} vs Round')
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_validation_metric_best_vs_round(records: list[RunRecord], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    objective = _metric_optimization_objective(metric)
    for record in records:
        rounds, values, _uploads = _validation_series(record, metric)
        best_values = _best_so_far_values(values, metric)
        if not rounds or not best_values:
            continue
        plt.plot(rounds, best_values, linewidth=1.8, label=record.label)
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, f'No runs with validation {METRIC_TITLES[metric]}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Round')
    plt.ylabel(f'Best-so-far Validation {METRIC_TITLES[metric]}')
    plt.title(f'Best-so-far Validation {METRIC_TITLES[metric]} vs Round ({objective})')
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_validation_metric_best_vs_upload(records: list[RunRecord], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    objective = _metric_optimization_objective(metric)
    for record in records:
        _rounds, values, uploads = _validation_series(record, metric)
        best_values = _best_so_far_values(values, metric)
        if not uploads or not best_values:
            continue
        if max(uploads, default=0) == 0:
            continue
        plt.plot(uploads, best_values, linewidth=1.8, label=record.label)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel(f'Best-so-far Validation {METRIC_TITLES[metric]}')
    plt.title(f'Best-so-far Validation {METRIC_TITLES[metric]} vs Cumulative Upload Bytes ({objective})')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_validation_metric_vs_upload(records: list[RunRecord], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for record in records:
        _rounds, values, uploads = _validation_series(record, metric)
        if not uploads or not values:
            continue
        if max(uploads, default=0) == 0:
            continue
        plt.plot(uploads, values, linewidth=1.8, label=record.label)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel(f'Validation {METRIC_TITLES[metric]}')
    plt.title(f'Validation {METRIC_TITLES[metric]} vs Cumulative Upload Bytes')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_test_metric_bar(rows: list[dict[str, Any]], metric: str, output_path: Path, title: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_key = _metric_key(metric)
    centralized_row = next((row for row in rows if row.get('label') == 'centralized' and row.get(metric_key) is not None), None)
    bar_rows = [row for row in rows if row.get('label') != 'centralized' and row.get(metric_key) is not None]
    labels = [str(row['label']) for row in bar_rows]
    values = [float(row[metric_key]) for row in bar_rows]
    colors = [_bar_color(str(row['label'])) for row in bar_rows]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if labels:
        ax.bar(labels, values, color=colors)
        plt.xticks(rotation=20, ha='right')
    elif centralized_row is None:
        ax.text(0.5, 0.5, f'No runs with test {METRIC_TITLES[metric]}', ha='center', va='center', transform=ax.transAxes)
    if centralized_row is not None:
        baseline = float(centralized_row[metric_key])
        ax.axhline(baseline, linestyle='--', color='#4c78a8', linewidth=1.8, label='centralized baseline')
        ax.legend()
    ax.set_ylabel(f'Test {METRIC_TITLES[metric]}')
    ax.set_title(title or f'Test {METRIC_TITLES[metric]}')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_test_metric_vs_upload_bubble(rows: list[dict[str, Any]], metric: str, output_path: Path, title: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metric_key = _metric_key(metric)
    plot_rows = [row for row in rows if row.get(metric_key) is not None and row.get('total_upload_bytes') is not None]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if not plot_rows:
        ax.text(0.5, 0.5, f'No runs with test {METRIC_TITLES[metric]} and upload communication', ha='center', va='center', transform=ax.transAxes)
    else:
        for row in plot_rows:
            label = str(row['label'])
            x_value = float(row['total_upload_bytes'])
            y_value = float(row[metric_key])
            ratio = row.get('fedavg_upload_compression_ratio')
            bubble_size = _bubble_size_from_ratio(float(ratio)) if ratio is not None else _bubble_size_from_ratio(None)
            ax.scatter(x_value, y_value, s=bubble_size, color=_bar_color(label), alpha=0.72, edgecolors='black', linewidths=0.8, label=label)
            ax.annotate(label, (x_value, y_value), textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xlabel('Total Upload Bytes')
    ax.set_ylabel(f'Test {METRIC_TITLES[metric]}')
    ax.set_title(title or f'Test {METRIC_TITLES[metric]} vs Upload Communication')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_validation_metric_vs_round(curves: list[AggregatedCurve], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    for curve in curves:
        if not curve.rounds or not curve.round_mean:
            continue
        lower = [avg - std for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.rounds, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.rounds, lower, upper, alpha=0.18)
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, f'No runs with validation {METRIC_TITLES[metric]}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Round')
    plt.ylabel(f'Validation {METRIC_TITLES[metric]}')
    plt.title(f'Validation {METRIC_TITLES[metric]} vs Round (mean +- std)')
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_validation_metric_best_vs_round(curves: list[AggregatedCurve], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted = False
    objective = _metric_optimization_objective(metric)
    for curve in curves:
        if not curve.rounds or not curve.round_mean:
            continue
        lower = [avg - std for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.rounds, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.rounds, lower, upper, alpha=0.18)
        plotted = True
    if not plotted:
        plt.text(0.5, 0.5, f'No runs with validation {METRIC_TITLES[metric]}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Round')
    plt.ylabel(f'Best-so-far Validation {METRIC_TITLES[metric]}')
    plt.title(f'Best-so-far Validation {METRIC_TITLES[metric]} vs Round (mean +- std, {objective})')
    plt.grid(True, alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_validation_metric_best_vs_upload(curves: list[AggregatedCurve], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    objective = _metric_optimization_objective(metric)
    for curve in curves:
        if not curve.upload_mean or not curve.round_mean:
            continue
        if max(curve.upload_mean, default=0.0) == 0.0:
            continue
        lower = [avg - std for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.upload_mean, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.upload_mean, lower, upper, alpha=0.18)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel(f'Best-so-far Validation {METRIC_TITLES[metric]}')
    plt.title(f'Best-so-far Validation {METRIC_TITLES[metric]} vs Cumulative Upload Bytes (mean +- std, {objective})')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_validation_metric_vs_upload(curves: list[AggregatedCurve], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for curve in curves:
        if not curve.upload_mean or not curve.round_mean:
            continue
        if max(curve.upload_mean, default=0.0) == 0.0:
            continue
        lower = [avg - std for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.upload_mean, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.upload_mean, lower, upper, alpha=0.18)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel(f'Validation {METRIC_TITLES[metric]}')
    plt.title(f'Validation {METRIC_TITLES[metric]} vs Cumulative Upload Bytes (mean +- std)')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_test_metric_bar(rows: list[dict[str, Any]], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_key = _metric_mean_key(metric)
    std_key = _metric_std_key(metric)
    centralized_row = next((row for row in rows if row.get('label') == 'centralized' and row.get(mean_key) is not None), None)
    bar_rows = [row for row in rows if row.get('label') != 'centralized' and row.get(mean_key) is not None]
    labels = [str(row['label']) for row in bar_rows]
    values = [float(row[mean_key]) for row in bar_rows]
    errors = [float(row.get(std_key) or 0.0) for row in bar_rows]
    colors = [_bar_color(str(row['label'])) for row in bar_rows]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if labels:
        ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
        plt.xticks(rotation=20, ha='right')
    elif centralized_row is None:
        ax.text(0.5, 0.5, f'No runs with test {METRIC_TITLES[metric]}', ha='center', va='center', transform=ax.transAxes)
    if centralized_row is not None:
        baseline = float(centralized_row[mean_key])
        baseline_std = float(centralized_row.get(std_key) or 0.0)
        ax.axhline(baseline, linestyle='--', color='#4c78a8', linewidth=1.8, label='centralized baseline')
        if baseline_std > 0.0:
            ax.axhspan(baseline - baseline_std, baseline + baseline_std, color='#4c78a8', alpha=0.12)
        ax.legend()
    ax.set_ylabel(f'Test {METRIC_TITLES[metric]}')
    ax.set_title(f'Test {METRIC_TITLES[metric]} (mean +- std)')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_aggregated_test_metric_vs_upload_bubble(rows: list[dict[str, Any]], metric: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mean_key = _metric_mean_key(metric)
    std_key = _metric_std_key(metric)
    plot_rows = [row for row in rows if row.get(mean_key) is not None and row.get('total_upload_bytes_mean') is not None]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if not plot_rows:
        ax.text(0.5, 0.5, f'No runs with test {METRIC_TITLES[metric]} and upload communication', ha='center', va='center', transform=ax.transAxes)
    else:
        for row in plot_rows:
            label = str(row['label'])
            x_value = float(row['total_upload_bytes_mean'])
            y_value = float(row[mean_key])
            x_error = float(row.get('total_upload_bytes_std') or 0.0)
            y_error = float(row.get(std_key) or 0.0)
            ratio = row.get('fedavg_upload_compression_ratio_mean')
            bubble_size = _bubble_size_from_ratio(float(ratio)) if ratio is not None else _bubble_size_from_ratio(None)
            ax.scatter(x_value, y_value, s=bubble_size, color=_bar_color(label), alpha=0.72, edgecolors='black', linewidths=0.8, label=label)
            if x_error > 0.0 or y_error > 0.0:
                ax.errorbar(x_value, y_value, xerr=x_error, yerr=y_error, fmt='none', ecolor=_bar_color(label), alpha=0.55, capsize=4)
            ax.annotate(label, (x_value, y_value), textcoords='offset points', xytext=(6, 6), fontsize=9)
    ax.set_xlabel('Total Upload Bytes')
    ax.set_ylabel(f'Test {METRIC_TITLES[metric]}')
    ax.set_title(f'Test {METRIC_TITLES[metric]} vs Upload Communication (mean +- std)')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
def plot_val_mse_vs_round(records: list[RunRecord], output_path: Path) -> Path:
    return plot_validation_metric_vs_round(records, 'mse', output_path)
def plot_val_mse_vs_upload(records: list[RunRecord], output_path: Path) -> Path:
    return plot_validation_metric_vs_upload(records, 'mse', output_path)
def plot_test_mse_bar(rows: list[dict[str, Any]], output_path: Path, title: str = 'Test MSE') -> Path:
    return plot_test_metric_bar(rows, 'mse', output_path, title=title)
def plot_test_mse_vs_upload_bubble(rows: list[dict[str, Any]], output_path: Path, title: str = 'Test MSE vs Upload Communication') -> Path:
    return plot_test_metric_vs_upload_bubble(rows, 'mse', output_path, title=title)
def plot_aggregated_val_mse_vs_round(curves: list[AggregatedCurve], output_path: Path) -> Path:
    return plot_aggregated_validation_metric_vs_round(curves, 'mse', output_path)
def plot_aggregated_val_mse_vs_upload(curves: list[AggregatedCurve], output_path: Path) -> Path:
    return plot_aggregated_validation_metric_vs_upload(curves, 'mse', output_path)
def plot_aggregated_test_mse_bar(rows: list[dict[str, Any]], output_path: Path) -> Path:
    return plot_aggregated_test_metric_bar(rows, 'mse', output_path)
def plot_aggregated_test_mse_vs_upload_bubble(rows: list[dict[str, Any]], output_path: Path) -> Path:
    return plot_aggregated_test_metric_vs_upload_bubble(rows, 'mse', output_path)
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize and plot selected experiment-suite results.')
    parser.add_argument('root_dir', nargs='+', type=Path, help='One or more suite root directories or one noattack_* subdirectory per seed')
    parser.add_argument('--loss', default='mse', help='Prefer one noattack_{loss} subdirectory when present; also accepts classification losses such as cross_entropy')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory used to save summary tables and plots; default is a sibling analysis directory')
    parser.add_argument('--include-old', action='store_true', help='Include *_old run directories in the summary and plots')
    parser.add_argument('--algorithms', nargs='*', default=list(DEFAULT_ALGORITHMS), help='Algorithm-name tokens used to select experiment directories by substring matching')
    parser.add_argument('--metrics', nargs='+', choices=SUPPORTED_METRICS, default=None, help='Validation/test metrics to summarize and plot; defaults to metrics inferred from the selected runs')
    return parser
def _stdout_metric_suffix(row: dict[str, Any], metrics: tuple[str, ...], aggregated: bool) -> str:
    parts: list[str] = []
    for metric in metrics:
        if aggregated:
            parts.append(f"{metric}={format_float(row.get(_metric_mean_key(metric)))}+-{format_float(row.get(_metric_std_key(metric)))}")
            parts.append(f"{metric}_delta={format_float(row.get(_metric_relative_mean_key(metric)))}+-{format_float(row.get(_metric_relative_std_key(metric)))}%")
        else:
            parts.append(f"{metric}={format_float(row.get(_metric_key(metric)))}")
            parts.append(f"{metric}_delta={format_float(row.get(_metric_relative_key(metric)))}%")
    return ' '.join(parts)
def summarize_single_suite(root_dir: Path, loss: str, algorithms: tuple[str, ...], include_old: bool, output_dir: Path, metrics: tuple[str, ...] | list[str] | None = None) -> None:
    records = discover_runs(root_dir, loss, algorithms, include_old)
    if not records:
        raise ValueError(f'No eligible runs found under {resolve_suite_dir(root_dir, loss)}')
    selected_metrics = _resolve_metrics(records, metrics)
    rows = build_rows(records, selected_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_summary_csv(rows, output_dir / 'summary.csv', selected_metrics)
    md_path = write_summary_markdown(rows, output_dir / 'summary.md', selected_metrics)
    print(f'Saved {csv_path}')
    print(f'Saved {md_path}')
    for metric in selected_metrics:
        round_plot = plot_validation_metric_vs_round(records, metric, output_dir / f'val_{metric}_vs_round.png')
        best_round_plot = plot_validation_metric_best_vs_round(records, metric, output_dir / f'val_{metric}_best_so_far_vs_round.png')
        upload_plot = plot_validation_metric_vs_upload(records, metric, output_dir / f'val_{metric}_vs_cumulative_upload.png')
        best_upload_plot = plot_validation_metric_best_vs_upload(records, metric, output_dir / f'val_{metric}_best_so_far_vs_cumulative_upload.png')
        test_plot = plot_test_metric_bar(rows, metric, output_dir / f'test_{metric}_bar.png')
        bubble_plot = plot_test_metric_vs_upload_bubble(rows, metric, output_dir / f'test_{metric}_vs_upload_bubble.png')
        print(f'Saved {round_plot}')
        print(f'Saved {best_round_plot}')
        print(f'Saved {upload_plot}')
        print(f'Saved {best_upload_plot}')
        print(f'Saved {test_plot}')
        print(f'Saved {bubble_plot}')
    for row in rows:
        attack_suffix = ''
        if row['attack_primary_metric_name'] is not None:
            attack_suffix = (
                f" attack_metric={row['attack_primary_metric_name']}"
                f" attack_avg={format_float(row['attack_overall_avg_primary_metric_value'])}"
                f" attack_success={format_float(row['attack_success_rate'])}"
                f" attack_evals={row['attack_evaluations']}"
            )
        print(
            f"{row['label']}: {_stdout_metric_suffix(row, selected_metrics, aggregated=False)} "
            f"upload={row['total_upload_bytes']} "
            f"fedavg/f={format_float(row['fedavg_upload_compression_ratio'])}"
            f"{attack_suffix}"
        )
def summarize_multi_suite(root_dirs: list[Path], loss: str, algorithms: tuple[str, ...], include_old: bool, output_dir: Path, metrics: tuple[str, ...] | list[str] | None = None) -> None:
    records_by_seed = [discover_runs(root_dir, loss, algorithms, include_old) for root_dir in root_dirs]
    if any(not records for records in records_by_seed):
        missing = [str(resolve_suite_dir(root_dir, loss)) for root_dir, records in zip(root_dirs, records_by_seed) if not records]
        raise ValueError(f'No eligible runs found under: {", ".join(missing)}')
    selected_metrics = _resolve_metrics([record for records in records_by_seed for record in records], metrics)
    rows = aggregate_rows(records_by_seed, selected_metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_aggregated_summary_csv(rows, output_dir / 'summary.csv', selected_metrics)
    md_path = write_aggregated_summary_markdown(rows, output_dir / 'summary.md', selected_metrics)
    print(f'Saved {csv_path}')
    print(f'Saved {md_path}')
    for metric in selected_metrics:
        curves = aggregate_metric_curves(records_by_seed, metric)
        best_curves = aggregate_best_metric_curves(records_by_seed, metric)
        round_plot = plot_aggregated_validation_metric_vs_round(curves, metric, output_dir / f'val_{metric}_vs_round.png')
        best_round_plot = plot_aggregated_validation_metric_best_vs_round(best_curves, metric, output_dir / f'val_{metric}_best_so_far_vs_round.png')
        upload_plot = plot_aggregated_validation_metric_vs_upload(curves, metric, output_dir / f'val_{metric}_vs_cumulative_upload.png')
        best_upload_plot = plot_aggregated_validation_metric_best_vs_upload(best_curves, metric, output_dir / f'val_{metric}_best_so_far_vs_cumulative_upload.png')
        test_plot = plot_aggregated_test_metric_bar(rows, metric, output_dir / f'test_{metric}_bar.png')
        bubble_plot = plot_aggregated_test_metric_vs_upload_bubble(rows, metric, output_dir / f'test_{metric}_vs_upload_bubble.png')
        print(f'Saved {round_plot}')
        print(f'Saved {best_round_plot}')
        print(f'Saved {upload_plot}')
        print(f'Saved {best_upload_plot}')
        print(f'Saved {test_plot}')
        print(f'Saved {bubble_plot}')
    for row in rows:
        attack_suffix = ''
        if row['attack_primary_metric_name'] is not None:
            attack_suffix = (
                f" attack_metric={row['attack_primary_metric_name']}"
                f" attack_avg={format_float(row['attack_overall_avg_primary_metric_value_mean'])}+-{format_float(row['attack_overall_avg_primary_metric_value_std'])}"
                f" attack_success={format_float(row['attack_success_rate_mean'])}+-{format_float(row['attack_success_rate_std'])}"
                f" attack_evals={format_float(row['attack_evaluations_mean'])}+-{format_float(row['attack_evaluations_std'])}"
            )
        print(
            f"{row['label']}: {_stdout_metric_suffix(row, selected_metrics, aggregated=True)} "
            f"upload={format_float(row['total_upload_bytes_mean'])}+-{format_float(row['total_upload_bytes_std'])} "
            f"fedavg/f={format_float(row['fedavg_upload_compression_ratio_mean'])}+-{format_float(row['fedavg_upload_compression_ratio_std'])}"
            f"{attack_suffix}"
        )
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    root_dirs = [path.resolve() for path in args.root_dir]
    for root_dir in root_dirs:
        if not root_dir.is_dir():
            raise NotADirectoryError(f'{root_dir} is not a directory')
    raw_algorithms = args.algorithms
    algorithms = normalize_algorithm_tokens(raw_algorithms)
    loss = args.loss
    metrics = _normalize_metric_tokens(args.metrics)
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif len(root_dirs) == 1:
        output_dir = default_output_dir(resolve_suite_dir(root_dirs[0], loss), loss)
    else:
        output_dir = default_multi_output_dir([resolve_suite_dir(root_dir, loss) for root_dir in root_dirs], loss)
    if len(root_dirs) == 1:
        summarize_single_suite(root_dirs[0], loss, algorithms, args.include_old, output_dir, metrics)
    else:
        summarize_multi_suite(root_dirs, loss, algorithms, args.include_old, output_dir, metrics)
if __name__ == '__main__':
    main()
