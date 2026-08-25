#!/usr/bin/env python3
"""Summarize and plot selected experiment-suite results.

The tool scans one or more suite directories, selects runs by algorithm-name
matching, then exports tabular summaries plus validation/test-MSE
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
)
DEFAULT_ALGORITHMS = ('centralized', 'fedavg', 'topk', 'ega')
BAR_COLORS = {
    'centralized': '#4c78a8',
    'topk': '#f58518',
    'ega': '#54a24b',
}


@dataclass
class RunRecord:
    label: str
    run_name: str
    run_dir: Path
    algorithm: str
    val_rounds: list[int]
    val_mse: list[float]
    val_cumulative_upload_bytes: list[int]
    test_mse: float | None
    total_upload_bytes: int
    attack_present: bool
    attack_primary_metric_name: str | None
    attack_primary_metric_direction: str | None
    attack_overall_avg_primary_metric_value: float | None
    attack_success_rate: float | None
    attack_evaluations: int | None


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
    candidate = root_dir / f'noattack_{loss}'
    return candidate if candidate.is_dir() else root_dir


def default_output_dir(root_dir: Path, loss: str | None) -> Path:
    suffix = loss or 'all'
    return root_dir.parent / f'{root_dir.name}_analysis_{suffix}'


def default_multi_output_dir(root_dirs: list[Path], loss: str | None) -> Path:
    suffix = loss or 'all'
    common_parent = root_dirs[0].parent
    return common_parent / f'multiseed_analysis_{suffix}'


def normalize_algorithm_tokens(algorithms: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(token.strip().lower() for token in algorithms if token.strip())


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


def _resolve_val_mse(record: dict[str, Any]) -> float | None:
    for key in ('active_val_mse', 'protocol_val_mse', 'val_mse'):
        if key in record and record[key] is not None:
            return float(record[key])
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
    if avg_primary_value is None and primary_metric_name == 'reconstruction_mse':
        avg_primary_value = summary.get('attack_overall_avg_mse')
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

    val_rounds: list[int] = []
    val_mse: list[float] = []
    val_cumulative_upload_bytes: list[int] = []

    if isinstance(metrics, dict):
        history = metrics.get('history') or []
        if not isinstance(history, list):
            raise TypeError(f'Unsupported centralized metrics format in {run_dir}')
        for idx, item in enumerate(history):
            value = _resolve_val_mse(item)
            if value is None:
                continue
            val_rounds.append(int(item.get('round', idx)))
            val_mse.append(value)
            val_cumulative_upload_bytes.append(0)
    elif isinstance(metrics, list):
        running = 0
        for idx, item in enumerate(metrics):
            running += _resolve_round_upload_bytes(item)
            value = _resolve_val_mse(item)
            if value is None:
                continue
            val_rounds.append(int(item.get('round', idx)))
            val_mse.append(value)
            val_cumulative_upload_bytes.append(running)
    else:
        raise TypeError(f'Unsupported metrics format in {run_dir}: {type(metrics)!r}')

    test_blob = summary.get('test') or summary.get('protocol_test') or {}
    test_mse = float(test_blob['mse']) if 'mse' in test_blob else None
    total_upload_bytes = int(summary.get('total_parameter_upload_bytes', summary.get('total_upload_bytes', 0)) or 0)
    algorithm = str(summary.get('algorithm') or summary.get('evaluation_mode') or 'unknown')
    attack_info = _resolve_attack_info(summary)

    return RunRecord(
        label=canonical_label(run_dir.name),
        run_name=run_dir.name,
        run_dir=run_dir,
        algorithm=algorithm,
        val_rounds=val_rounds,
        val_mse=val_mse,
        val_cumulative_upload_bytes=val_cumulative_upload_bytes,
        test_mse=test_mse,
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


def build_rows(records: list[RunRecord]) -> list[dict[str, Any]]:
    fedavg = next((record for record in records if record.label == 'fedavg'), None)
    if fedavg is None:
        raise ValueError('FedAvg run is required to compute compression and loss ratios')
    if fedavg.test_mse is None:
        raise ValueError('FedAvg summary is missing test.mse')

    rows = []
    for record in records:
        compression_ratio = None
        if record.total_upload_bytes > 0:
            compression_ratio = fedavg.total_upload_bytes / record.total_upload_bytes
        mse_loss_ratio_percent = None
        if record.test_mse is not None:
            mse_loss_ratio_percent = (record.test_mse - fedavg.test_mse) / fedavg.test_mse * 100.0
        rows.append({
            'label': record.label,
            'run_name': record.run_name,
            'test_mse': record.test_mse,
            'total_upload_bytes': record.total_upload_bytes,
            'fedavg_upload_compression_ratio': compression_ratio,
            'mse_loss_ratio_percent': mse_loss_ratio_percent,
            'attack_primary_metric_name': record.attack_primary_metric_name if record.attack_present else None,
            'attack_primary_metric_direction': record.attack_primary_metric_direction if record.attack_present else None,
            'attack_overall_avg_primary_metric_value': record.attack_overall_avg_primary_metric_value if record.attack_present else None,
            'attack_success_rate': record.attack_success_rate if record.attack_present else None,
            'attack_evaluations': record.attack_evaluations if record.attack_present else None,
        })
    return rows


def aggregate_rows(records_by_seed: list[list[RunRecord]]) -> list[dict[str, Any]]:
    rows_by_seed = [build_rows(records) for records in records_by_seed]
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
        test_values = [float(entry['test_mse']) for entry in entries if entry['test_mse'] is not None]
        upload_values = [float(entry['total_upload_bytes']) for entry in entries if entry['total_upload_bytes'] is not None]
        ratio_values = [float(entry['fedavg_upload_compression_ratio']) for entry in entries if entry['fedavg_upload_compression_ratio'] is not None]
        loss_values = [float(entry['mse_loss_ratio_percent']) for entry in entries if entry['mse_loss_ratio_percent'] is not None]
        attack_metric_values = [float(entry['attack_overall_avg_primary_metric_value']) for entry in entries if entry['attack_overall_avg_primary_metric_value'] is not None]
        attack_success_values = [float(entry['attack_success_rate']) for entry in entries if entry['attack_success_rate'] is not None]
        attack_eval_values = [float(entry['attack_evaluations']) for entry in entries if entry['attack_evaluations'] is not None]
        test_mean, test_std = _mean_std(test_values)
        upload_mean, upload_std = _mean_std(upload_values)
        ratio_mean, ratio_std = _mean_std(ratio_values)
        loss_mean, loss_std = _mean_std(loss_values)
        attack_metric_mean, attack_metric_std = _mean_std(attack_metric_values)
        attack_success_mean, attack_success_std = _mean_std(attack_success_values)
        attack_eval_mean, attack_eval_std = _mean_std(attack_eval_values)
        aggregated_rows.append({
            'label': label,
            'seed_count': len(entries),
            'test_mse_mean': test_mean,
            'test_mse_std': test_std,
            'total_upload_bytes_mean': upload_mean,
            'total_upload_bytes_std': upload_std,
            'fedavg_upload_compression_ratio_mean': ratio_mean,
            'fedavg_upload_compression_ratio_std': ratio_std,
            'mse_loss_ratio_percent_mean': loss_mean,
            'mse_loss_ratio_percent_std': loss_std,
            'attack_seed_count': len(attack_metric_values) or len(attack_success_values) or len(attack_eval_values),
            'attack_primary_metric_name': _unique_or_mixed([entry['attack_primary_metric_name'] for entry in entries]),
            'attack_primary_metric_direction': _unique_or_mixed([entry['attack_primary_metric_direction'] for entry in entries]),
            'attack_overall_avg_primary_metric_value_mean': attack_metric_mean,
            'attack_overall_avg_primary_metric_value_std': attack_metric_std,
            'attack_success_rate_mean': attack_success_mean,
            'attack_success_rate_std': attack_success_std,
            'attack_evaluations_mean': attack_eval_mean,
            'attack_evaluations_std': attack_eval_std,
        })
    return aggregated_rows


def aggregate_curves(records_by_seed: list[list[RunRecord]]) -> list[AggregatedCurve]:
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
            for round_id, value, upload in zip(record.val_rounds, record.val_mse, record.val_cumulative_upload_bytes):
                round_to_val[round_id].append(value)
                round_to_upload[round_id].append(float(upload))
        rounds = sorted(round_to_val.keys())
        round_mean: list[float] = []
        round_std: list[float] = []
        upload_mean: list[float] = []
        upload_std: list[float] = []
        for round_id in rounds:
            value_mean, value_std = _mean_std(round_to_val[round_id])
            upload_mean_value, upload_std_value = _mean_std(round_to_upload[round_id])
            if value_mean is None or value_std is None or upload_mean_value is None or upload_std_value is None:
                continue
            round_mean.append(value_mean)
            round_std.append(value_std)
            upload_mean.append(upload_mean_value)
            upload_std.append(upload_std_value)
        curves.append(AggregatedCurve(
            label=label,
            rounds=rounds[: len(round_mean)],
            round_mean=round_mean,
            round_std=round_std,
            upload_mean=upload_mean,
            upload_std=upload_std,
        ))
    return curves


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return 'n/a'
    return f'{value:.{digits}f}'


def format_bytes(value: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    current = float(value)
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f'{current:.2f}{unit}'
        current /= 1024.0
    return f'{current:.2f}TB'


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'label',
        'run_name',
        'test_mse',
        'total_upload_bytes',
        'fedavg_upload_compression_ratio',
        'mse_loss_ratio_percent',
        'attack_primary_metric_name',
        'attack_primary_metric_direction',
        'attack_overall_avg_primary_metric_value',
        'attack_success_rate',
        'attack_evaluations',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_aggregated_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'label',
        'seed_count',
        'test_mse_mean',
        'test_mse_std',
        'total_upload_bytes_mean',
        'total_upload_bytes_std',
        'fedavg_upload_compression_ratio_mean',
        'fedavg_upload_compression_ratio_std',
        'mse_loss_ratio_percent_mean',
        'mse_loss_ratio_percent_std',
        'attack_seed_count',
        'attack_primary_metric_name',
        'attack_primary_metric_direction',
        'attack_overall_avg_primary_metric_value_mean',
        'attack_overall_avg_primary_metric_value_std',
        'attack_success_rate_mean',
        'attack_success_rate_std',
        'attack_evaluations_mean',
        'attack_evaluations_std',
    ]
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_summary_markdown(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '| Label | Test MSE | Total Upload | FedAvg/F | MSE Loss % | Attack Metric | Attack Avg | Attack Success | Attack Evals |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            '| {label} | {mse} | {upload} | {ratio} | {loss} | {attack_metric} | {attack_avg} | {attack_success} | {attack_evals} |'.format(
                label=row['label'],
                mse=format_float(row['test_mse']),
                upload=f"{row['total_upload_bytes']} ({format_bytes(int(row['total_upload_bytes']))})",
                ratio=format_float(row['fedavg_upload_compression_ratio']),
                loss=format_float(row['mse_loss_ratio_percent']),
                attack_metric=row['attack_primary_metric_name'] or 'n/a',
                attack_avg=format_float(row['attack_overall_avg_primary_metric_value']),
                attack_success=format_float(row['attack_success_rate']),
                attack_evals=str(row['attack_evaluations']) if row['attack_evaluations'] is not None else 'n/a',
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path


def write_aggregated_summary_markdown(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '| Label | Seeds | Test MSE | Total Upload | FedAvg/F | MSE Loss % | Attack Metric | Attack Seeds | Attack Avg | Attack Success | Attack Evals |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        upload_mean = int(round(row['total_upload_bytes_mean'])) if row['total_upload_bytes_mean'] is not None else None
        lines.append(
            '| {label} | {seed_count} | {test} +- {test_std} | {upload} +- {upload_std} | {ratio} +- {ratio_std} | {loss} +- {loss_std} | {attack_metric} | {attack_seed_count} | {attack_avg} +- {attack_avg_std} | {attack_success} +- {attack_success_std} | {attack_evals} +- {attack_evals_std} |'.format(
                label=row['label'],
                seed_count=row['seed_count'],
                test=format_float(row['test_mse_mean']),
                test_std=format_float(row['test_mse_std']),
                upload=(f"{upload_mean} ({format_bytes(upload_mean)})" if upload_mean is not None else 'n/a'),
                upload_std=format_float(row['total_upload_bytes_std']),
                ratio=format_float(row['fedavg_upload_compression_ratio_mean']),
                ratio_std=format_float(row['fedavg_upload_compression_ratio_std']),
                loss=format_float(row['mse_loss_ratio_percent_mean']),
                loss_std=format_float(row['mse_loss_ratio_percent_std']),
                attack_metric=row['attack_primary_metric_name'] or 'n/a',
                attack_seed_count=row['attack_seed_count'],
                attack_avg=format_float(row['attack_overall_avg_primary_metric_value_mean']),
                attack_avg_std=format_float(row['attack_overall_avg_primary_metric_value_std']),
                attack_success=format_float(row['attack_success_rate_mean']),
                attack_success_std=format_float(row['attack_success_rate_std']),
                attack_evals=format_float(row['attack_evaluations_mean']),
                attack_evals_std=format_float(row['attack_evaluations_std']),
            )
        )
    output_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output_path


def plot_val_mse_vs_round(records: list[RunRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for record in records:
        if not record.val_rounds or not record.val_mse:
            continue
        plt.plot(record.val_rounds, record.val_mse, linewidth=1.8, label=record.label)
    plt.xlabel('Round')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Round')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_val_mse_vs_upload(records: list[RunRecord], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for record in records:
        if not record.val_cumulative_upload_bytes or not record.val_mse:
            continue
        if max(record.val_cumulative_upload_bytes, default=0) == 0:
            continue
        plt.plot(record.val_cumulative_upload_bytes, record.val_mse, linewidth=1.8, label=record.label)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Cumulative Upload Bytes')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_test_mse_bar(rows: list[dict[str, Any]], output_path: Path, title: str = 'Test MSE') -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fedavg_row = next((row for row in rows if row.get('label') == 'fedavg' and row.get('test_mse') is not None), None)
    bar_rows = [row for row in rows if row.get('label') != 'fedavg' and row.get('test_mse') is not None]
    labels = [str(row['label']) for row in bar_rows]
    values = [float(row['test_mse']) for row in bar_rows]
    colors = [_bar_color(str(row['label'])) for row in bar_rows]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if labels:
        ax.bar(labels, values, color=colors)
        plt.xticks(rotation=20, ha='right')
    elif fedavg_row is None:
        ax.text(0.5, 0.5, 'No runs with test MSE', ha='center', va='center', transform=ax.transAxes)
    if fedavg_row is not None:
        baseline = float(fedavg_row['test_mse'])
        ax.axhline(baseline, linestyle='--', color='#e45756', linewidth=1.8, label='fedavg baseline')
        ax.legend()
    ax.set_ylabel('Test MSE')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_aggregated_val_mse_vs_round(curves: list[AggregatedCurve], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    for curve in curves:
        if not curve.rounds or not curve.round_mean:
            continue
        lower = [max(0.0, avg - std) for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.rounds, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.rounds, lower, upper, alpha=0.18)
    plt.xlabel('Round')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Round (mean +- std)')
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_aggregated_val_mse_vs_upload(curves: list[AggregatedCurve], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plotted_any = False
    for curve in curves:
        if not curve.upload_mean or not curve.round_mean:
            continue
        if max(curve.upload_mean, default=0.0) == 0.0:
            continue
        lower = [max(0.0, avg - std) for avg, std in zip(curve.round_mean, curve.round_std)]
        upper = [avg + std for avg, std in zip(curve.round_mean, curve.round_std)]
        plt.plot(curve.upload_mean, curve.round_mean, linewidth=1.8, label=curve.label)
        plt.fill_between(curve.upload_mean, lower, upper, alpha=0.18)
        plotted_any = True
    if not plotted_any:
        plt.text(0.5, 0.5, 'No runs with upload communication history', ha='center', va='center', transform=plt.gca().transAxes)
    plt.xlabel('Cumulative Upload Bytes')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Cumulative Upload Bytes (mean +- std)')
    plt.grid(True, alpha=0.25)
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_aggregated_test_mse_bar(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fedavg_row = next((row for row in rows if row.get('label') == 'fedavg' and row.get('test_mse_mean') is not None), None)
    bar_rows = [row for row in rows if row.get('label') != 'fedavg' and row.get('test_mse_mean') is not None]
    labels = [str(row['label']) for row in bar_rows]
    values = [float(row['test_mse_mean']) for row in bar_rows]
    errors = [float(row['test_mse_std'] or 0.0) for row in bar_rows]
    colors = [_bar_color(str(row['label'])) for row in bar_rows]

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    if labels:
        ax.bar(labels, values, yerr=errors, capsize=5, color=colors)
        plt.xticks(rotation=20, ha='right')
    elif fedavg_row is None:
        ax.text(0.5, 0.5, 'No runs with test MSE', ha='center', va='center', transform=ax.transAxes)
    if fedavg_row is not None:
        baseline = float(fedavg_row['test_mse_mean'])
        baseline_std = float(fedavg_row['test_mse_std'] or 0.0)
        ax.axhline(baseline, linestyle='--', color='#e45756', linewidth=1.8, label='fedavg baseline')
        if baseline_std > 0.0:
            ax.axhspan(baseline - baseline_std, baseline + baseline_std, color='#e45756', alpha=0.12)
        ax.legend()
    ax.set_ylabel('Test MSE')
    ax.set_title('Test MSE (mean +- std)')
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize and plot selected experiment-suite results.')
    parser.add_argument('root_dir', nargs='+', type=Path, help='One or more suite root directories or one noattack_* subdirectory per seed')
    parser.add_argument('--loss', choices=('mse', 'mae'), default='mse', help='Prefer one noattack_{loss} subdirectory when present')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory used to save summary tables and plots; default is a sibling analysis directory')
    parser.add_argument('--include-old', action='store_true', help='Include *_old run directories in the summary and plots')
    parser.add_argument('--algorithms', nargs='*', default=list(DEFAULT_ALGORITHMS), help='Algorithm-name tokens used to select experiment directories by substring matching')
    parser.add_argument('--prefixes', nargs='*', default=None, help='Deprecated alias for --algorithms')
    return parser


def summarize_single_suite(root_dir: Path, loss: str, algorithms: tuple[str, ...], include_old: bool, output_dir: Path) -> None:
    records = discover_runs(root_dir, loss, algorithms, include_old)
    if not records:
        raise ValueError(f'No eligible runs found under {resolve_suite_dir(root_dir, loss)}')
    rows = build_rows(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_summary_csv(rows, output_dir / 'summary.csv')
    md_path = write_summary_markdown(rows, output_dir / 'summary.md')
    round_plot = plot_val_mse_vs_round(records, output_dir / 'val_mse_vs_round.png')
    upload_plot = plot_val_mse_vs_upload(records, output_dir / 'val_mse_vs_cumulative_upload.png')
    test_plot = plot_test_mse_bar(rows, output_dir / 'test_mse_bar.png')

    print(f'Saved {csv_path}')
    print(f'Saved {md_path}')
    print(f'Saved {round_plot}')
    print(f'Saved {upload_plot}')
    print(f'Saved {test_plot}')
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
            f"{row['label']}: mse={format_float(row['test_mse'])} "
            f"upload={row['total_upload_bytes']} "
            f"fedavg/f={format_float(row['fedavg_upload_compression_ratio'])} "
            f"mse_loss={format_float(row['mse_loss_ratio_percent'])}%"
            f"{attack_suffix}"
        )


def summarize_multi_suite(root_dirs: list[Path], loss: str, algorithms: tuple[str, ...], include_old: bool, output_dir: Path) -> None:
    records_by_seed = [discover_runs(root_dir, loss, algorithms, include_old) for root_dir in root_dirs]
    if any(not records for records in records_by_seed):
        missing = [str(resolve_suite_dir(root_dir, loss)) for root_dir, records in zip(root_dirs, records_by_seed) if not records]
        raise ValueError(f'No eligible runs found under: {", ".join(missing)}')
    rows = aggregate_rows(records_by_seed)
    curves = aggregate_curves(records_by_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = write_aggregated_summary_csv(rows, output_dir / 'summary.csv')
    md_path = write_aggregated_summary_markdown(rows, output_dir / 'summary.md')
    round_plot = plot_aggregated_val_mse_vs_round(curves, output_dir / 'val_mse_vs_round.png')
    upload_plot = plot_aggregated_val_mse_vs_upload(curves, output_dir / 'val_mse_vs_cumulative_upload.png')
    test_plot = plot_aggregated_test_mse_bar(rows, output_dir / 'test_mse_bar.png')

    print(f'Saved {csv_path}')
    print(f'Saved {md_path}')
    print(f'Saved {round_plot}')
    print(f'Saved {upload_plot}')
    print(f'Saved {test_plot}')
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
            f"{row['label']}: mse={format_float(row['test_mse_mean'])}+-{format_float(row['test_mse_std'])} "
            f"upload={format_float(row['total_upload_bytes_mean'])}+-{format_float(row['total_upload_bytes_std'])} "
            f"fedavg/f={format_float(row['fedavg_upload_compression_ratio_mean'])}+-{format_float(row['fedavg_upload_compression_ratio_std'])} "
            f"mse_loss={format_float(row['mse_loss_ratio_percent_mean'])}+-{format_float(row['mse_loss_ratio_percent_std'])}%"
            f"{attack_suffix}"
        )


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    root_dirs = [path.resolve() for path in args.root_dir]
    for root_dir in root_dirs:
        if not root_dir.is_dir():
            raise NotADirectoryError(f'{root_dir} is not a directory')

    raw_algorithms = args.algorithms if args.prefixes is None else args.prefixes
    algorithms = normalize_algorithm_tokens(raw_algorithms)
    loss = args.loss
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    elif len(root_dirs) == 1:
        output_dir = default_output_dir(resolve_suite_dir(root_dirs[0], loss), loss)
    else:
        output_dir = default_multi_output_dir([resolve_suite_dir(root_dir, loss) for root_dir in root_dirs], loss)

    if len(root_dirs) == 1:
        summarize_single_suite(root_dirs[0], loss, algorithms, args.include_old, output_dir)
    else:
        summarize_multi_suite(root_dirs, loss, algorithms, args.include_old, output_dir)


if __name__ == '__main__':
    main()
