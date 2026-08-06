#!/usr/bin/env python3
"""Compare communication and performance gaps between two experiment folders.

The first directory is treated as the FedAvg baseline. The second directory is
reported as the candidate method. Communication totals are computed from the
full ``metrics.json`` history, while performance gaps are computed from
``summary.json`` test metrics.

Example:
    python -m federated_ts.tools.compare_experiment_results         outputs/test_payload_rerun_pat50/fedavg_seed2026_pat50_payloadv1         outputs/test_payload_rerun_pat50/secure_qint8_bidir_seed2026_pat50_payloadv1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_COMPARE_FILES = (
    'summary.json',
    'metrics.json',
    'config.yaml',
    'config.json',
    'run.log',
)
LOWER_IS_BETTER_METRICS = ('mse', 'mae', 'mape')


def load_json(path: Path) -> Any:
    """Load one JSON artifact file.

    Example:
        ``summary = load_json(run_dir / 'summary.json')``
    """

    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)



def common_files(dir_a: Path, dir_b: Path, compare_files: tuple[str, ...]) -> list[str]:
    """Return filenames that exist in both experiment directories."""

    return [name for name in compare_files if (dir_a / name).exists() and (dir_b / name).exists()]



def copy_common_files(dir_a: Path, dir_b: Path, target_dir: Path, files: list[str]) -> None:
    """Copy common files from both runs into one comparison directory."""

    target_a = target_dir / dir_a.name
    target_b = target_dir / dir_b.name
    target_a.mkdir(parents=True, exist_ok=True)
    target_b.mkdir(parents=True, exist_ok=True)
    for name in files:
        shutil.copy2(dir_a / name, target_a / name)
        shutil.copy2(dir_b / name, target_b / name)



def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Summarize one experiment directory from ``metrics.json`` and ``summary.json``.

    Example:
        ``summary = summarize_run(Path('outputs/my_run'))``
    """

    metrics_path = run_dir / 'metrics.json'
    summary_path = run_dir / 'summary.json'
    if not metrics_path.exists():
        raise FileNotFoundError(f'Missing metrics.json in {run_dir}')
    history = load_json(metrics_path)
    if not isinstance(history, list) or not history:
        raise ValueError(f'metrics.json in {run_dir} must be a non-empty round list')

    total_parameter_upload_bytes = int(sum(int(record.get('total_parameter_upload_bytes', record.get('total_upload_bytes', 0))) for record in history))
    total_parameter_download_bytes = int(sum(int(record.get('total_parameter_download_bytes', record.get('total_download_bytes', 0))) for record in history))
    total_transport_upload_bytes = int(sum(int(record.get('total_transport_upload_bytes', record.get('total_parameter_upload_bytes', record.get('total_upload_bytes', 0)))) for record in history))
    total_transport_download_bytes = int(sum(int(record.get('total_transport_download_bytes', record.get('total_parameter_download_bytes', record.get('total_download_bytes', 0)))) for record in history))
    fedavg_reference_upload_bytes = int(sum(int(record.get('fedavg_reference_upload_bytes', 0)) for record in history))
    fedavg_reference_total_bytes = int(sum(int(record.get('fedavg_reference_total_bytes', 0)) for record in history))
    total_parameter_bytes = total_parameter_upload_bytes + total_parameter_download_bytes
    total_transport_bytes = total_transport_upload_bytes + total_transport_download_bytes

    result: dict[str, Any] = {
        'run_dir': str(run_dir),
        'run_name': run_dir.name,
        'algorithm': history[-1].get('algorithm', 'unknown'),
        'rounds': len(history),
        'total_upload_bytes': total_parameter_upload_bytes,
        'total_download_bytes': total_parameter_download_bytes,
        'total_actual_bytes': total_transport_bytes,
        'total_parameter_upload_bytes': total_parameter_upload_bytes,
        'total_parameter_download_bytes': total_parameter_download_bytes,
        'total_parameter_bytes': total_parameter_bytes,
        'total_transport_upload_bytes': total_transport_upload_bytes,
        'total_transport_download_bytes': total_transport_download_bytes,
        'total_transport_bytes': total_transport_bytes,
        'fedavg_reference_upload_bytes': fedavg_reference_upload_bytes,
        'fedavg_reference_total_bytes': fedavg_reference_total_bytes,
        'total_upload_ratio': fedavg_reference_upload_bytes / max(total_parameter_upload_bytes, 1),
        'total_communication_ratio': fedavg_reference_total_bytes / max(total_parameter_bytes, 1),
        'transport_total_upload_ratio': fedavg_reference_upload_bytes / max(total_transport_upload_bytes, 1),
        'transport_total_communication_ratio': fedavg_reference_total_bytes / max(total_transport_bytes, 1),
        'test': {},
    }

    if summary_path.exists():
        summary = load_json(summary_path)
        if isinstance(summary, dict):
            result['test'] = summary.get('test', {}) or {}
            if 'rounds' in summary:
                result['summary_rounds'] = summary['rounds']
            if 'epochs' in summary:
                result['summary_epochs'] = summary['epochs']
    return result



def compute_performance_gaps(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute absolute and relative test-metric gaps from baseline to candidate.

    Relative change uses ``(candidate - baseline) / baseline * 100``. For these
    forecasting metrics, negative values mean the candidate improved.
    """

    baseline_test = baseline.get('test') or {}
    candidate_test = candidate.get('test') or {}
    gaps: dict[str, dict[str, float]] = {}
    for key in LOWER_IS_BETTER_METRICS:
        if key not in baseline_test or key not in candidate_test:
            continue
        base_value = float(baseline_test[key])
        cand_value = float(candidate_test[key])
        absolute_delta = cand_value - base_value
        relative_percent = (absolute_delta / base_value * 100.0) if base_value != 0 else float('inf')
        gaps[key] = {
            'baseline': base_value,
            'candidate': cand_value,
            'absolute_delta': absolute_delta,
            'relative_percent': relative_percent,
        }
    return gaps



def compute_communication_gaps(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    """Compute cross-run communication gaps with the first run as baseline."""

    base_parameter = int(baseline['total_parameter_bytes'])
    cand_parameter = int(candidate['total_parameter_bytes'])
    parameter_ratio = cand_parameter / max(base_parameter, 1)
    base_transport = int(baseline['total_transport_bytes'])
    cand_transport = int(candidate['total_transport_bytes'])
    transport_ratio = cand_transport / max(base_transport, 1)
    return {
        'baseline_total_parameter_bytes': float(base_parameter),
        'candidate_total_parameter_bytes': float(cand_parameter),
        'candidate_vs_baseline_parameter_ratio': parameter_ratio,
        'candidate_parameter_reduction_percent': (1.0 - parameter_ratio) * 100.0,
        'baseline_total_transport_bytes': float(base_transport),
        'candidate_total_transport_bytes': float(cand_transport),
        'candidate_vs_baseline_transport_ratio': transport_ratio,
        'candidate_transport_reduction_percent': (1.0 - transport_ratio) * 100.0,
        'candidate_vs_baseline_actual_ratio': transport_ratio,
        'candidate_actual_reduction_percent': (1.0 - transport_ratio) * 100.0,
        'baseline_total_communication_ratio': float(baseline['total_communication_ratio']),
        'candidate_total_communication_ratio': float(candidate['total_communication_ratio']),
        'baseline_total_upload_ratio': float(baseline['total_upload_ratio']),
        'candidate_total_upload_ratio': float(candidate['total_upload_ratio']),
        'baseline_transport_total_communication_ratio': float(baseline['transport_total_communication_ratio']),
        'candidate_transport_total_communication_ratio': float(candidate['transport_total_communication_ratio']),
        'baseline_transport_total_upload_ratio': float(baseline['transport_total_upload_ratio']),
        'candidate_transport_total_upload_ratio': float(candidate['transport_total_upload_ratio']),
    }



def format_bytes(num_bytes: int | float) -> str:
    """Format bytes into a readable unit string."""

    units = ['B', 'KB', 'MB', 'GB', 'TB']
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f'{value:.2f}{unit}'
        value /= 1024.0
    return f'{value:.2f}TB'



def print_run_summary(title: str, summary: dict[str, Any]) -> None:
    """Print one run summary block."""

    print(f'[{title}] {summary["run_name"]}')
    print(f'  algorithm: {summary["algorithm"]}')
    print(f'  rounds(metrics): {summary["rounds"]}')
    if 'summary_rounds' in summary:
        print(f'  rounds(summary): {summary["summary_rounds"]}')
    if 'summary_epochs' in summary:
        print(f'  epochs(summary): {summary["summary_epochs"]}')
    print(f'  total_parameter_upload_bytes: {summary["total_parameter_upload_bytes"]} ({format_bytes(summary["total_parameter_upload_bytes"])})')
    print(f'  total_parameter_download_bytes: {summary["total_parameter_download_bytes"]} ({format_bytes(summary["total_parameter_download_bytes"])})')
    print(f'  total_parameter_bytes: {summary["total_parameter_bytes"]} ({format_bytes(summary["total_parameter_bytes"])})')
    print(f'  total_transport_upload_bytes: {summary["total_transport_upload_bytes"]} ({format_bytes(summary["total_transport_upload_bytes"])})')
    print(f'  total_transport_download_bytes: {summary["total_transport_download_bytes"]} ({format_bytes(summary["total_transport_download_bytes"])})')
    print(f'  total_transport_bytes: {summary["total_transport_bytes"]} ({format_bytes(summary["total_transport_bytes"])})')
    print(f'  fedavg_reference_total_bytes: {summary["fedavg_reference_total_bytes"]} ({format_bytes(summary["fedavg_reference_total_bytes"])})')
    print(f'  total_upload_ratio: {summary["total_upload_ratio"]:.6f}')
    print(f'  total_communication_ratio: {summary["total_communication_ratio"]:.6f}')
    print(f'  transport_total_upload_ratio: {summary["transport_total_upload_ratio"]:.6f}')
    print(f'  transport_total_communication_ratio: {summary["transport_total_communication_ratio"]:.6f}')
    test = summary.get('test') or {}
    if test:
        print(f"  test: mse={test.get('mse', 'n/a')} mae={test.get('mae', 'n/a')} mape={test.get('mape', 'n/a')}")



def build_argparser() -> argparse.ArgumentParser:
    """Build the command line parser."""

    parser = argparse.ArgumentParser(description='Compare communication and performance between a FedAvg baseline and a candidate run.')
    parser.add_argument('baseline_dir', type=Path, help='FedAvg baseline experiment directory')
    parser.add_argument('candidate_dir', type=Path, help='Candidate method experiment directory')
    parser.add_argument(
        '--extract-dir',
        type=Path,
        default=None,
        help='Optional output directory used to copy common artifact files from both runs.',
    )
    parser.add_argument(
        '--compare-files',
        nargs='*',
        default=list(DEFAULT_COMPARE_FILES),
        help='Filenames to check and optionally extract from both directories.',
    )
    return parser



def main() -> None:
    """Run the experiment comparison CLI."""

    parser = build_argparser()
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    candidate_dir = args.candidate_dir.resolve()
    if not baseline_dir.is_dir():
        raise NotADirectoryError(f'{baseline_dir} is not a directory')
    if not candidate_dir.is_dir():
        raise NotADirectoryError(f'{candidate_dir} is not a directory')

    compare_files = tuple(args.compare_files)
    shared_files = common_files(baseline_dir, candidate_dir, compare_files)
    baseline = summarize_run(baseline_dir)
    candidate = summarize_run(candidate_dir)
    performance_gaps = compute_performance_gaps(baseline, candidate)
    communication_gaps = compute_communication_gaps(baseline, candidate)

    print('Common files:')
    if shared_files:
        for name in shared_files:
            print(f'  - {name}')
    else:
        print('  (none)')
    print()

    print_run_summary('FedAvg baseline', baseline)
    print()
    print_run_summary('Our method', candidate)
    print()

    print('Communication comparison:')
    print(f"  candidate_vs_baseline_parameter_ratio: {communication_gaps['candidate_vs_baseline_parameter_ratio']:.6f}")
    print(f"  candidate_parameter_reduction_percent: {communication_gaps['candidate_parameter_reduction_percent']:.2f}%")
    print(f"  baseline_total_upload_ratio: {communication_gaps['baseline_total_upload_ratio']:.6f}")
    print(f"  candidate_total_upload_ratio: {communication_gaps['candidate_total_upload_ratio']:.6f}")
    print(f"  baseline_total_communication_ratio: {communication_gaps['baseline_total_communication_ratio']:.6f}")
    print(f"  candidate_total_communication_ratio: {communication_gaps['candidate_total_communication_ratio']:.6f}")
    print(f"  candidate_vs_baseline_transport_ratio: {communication_gaps['candidate_vs_baseline_transport_ratio']:.6f}")
    print(f"  candidate_transport_reduction_percent: {communication_gaps['candidate_transport_reduction_percent']:.2f}%")
    print(f"  baseline_transport_total_upload_ratio: {communication_gaps['baseline_transport_total_upload_ratio']:.6f}")
    print(f"  candidate_transport_total_upload_ratio: {communication_gaps['candidate_transport_total_upload_ratio']:.6f}")
    print(f"  baseline_transport_total_communication_ratio: {communication_gaps['baseline_transport_total_communication_ratio']:.6f}")
    print(f"  candidate_transport_total_communication_ratio: {communication_gaps['candidate_transport_total_communication_ratio']:.6f}")
    print()

    print('Performance gaps (candidate - baseline, lower is better):')
    if performance_gaps:
        for key, values in performance_gaps.items():
            print(
                f"  {key}: absolute_delta={values['absolute_delta']:.12f} "
                f"relative_percent={values['relative_percent']:.6f}%"
            )
    else:
        print('  (no shared test metrics found)')

    if args.extract_dir is not None:
        target_dir = args.extract_dir.resolve()
        copy_common_files(baseline_dir, candidate_dir, target_dir, shared_files)
        print()
        print(f'extracted_common_files_to: {target_dir}')


if __name__ == '__main__':
    main()
