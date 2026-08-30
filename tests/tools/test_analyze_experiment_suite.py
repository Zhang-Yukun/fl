import csv
import importlib.util
import json
import math
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[2] / "fedlab" / "tools" / "analyze_experiment_suite.py"
spec = importlib.util.spec_from_file_location("analyze_experiment_suite", SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _attack_payload(avg_metric: float, success_rate: float, evaluations: int) -> dict:
    return {
        'attack_target_type': 'update_payload',
        'attack_primary_metric_name': 'budget_recovered_fraction',
        'attack_primary_metric_direction': 'lower_is_more_private',
        'attack_overall_avg_primary_metric_value': avg_metric,
        'attack_success_rate': success_rate,
        'attack_evaluations': evaluations,
        'attack_summary': {
            'primary_metric_name': 'budget_recovered_fraction',
            'primary_metric_direction': 'lower_is_more_private',
            'overall_avg_budget_recovered_fraction': avg_metric,
            'overall_success_rate': success_rate,
        },
    }


def _create_seed_suite(root: Path, seed_name: str, *, fedavg_test: float, topk_test: float, ega_test: float, fedavg_attack: tuple[float, float, int], topk_attack: tuple[float, float, int], ega_attack: tuple[float, float, int]) -> Path:
    suite_dir = root / seed_name / 'attack_mse'
    suite_dir.mkdir(parents=True)

    centralized_dir = suite_dir / 'centralized_attackfreq5'
    fedavg_dir = suite_dir / 'fedavg_single_sync_attackfreq5'
    topk_dir = suite_dir / 'topk_single_sync_attackfreq5'
    ega_dir = suite_dir / 'ega_ed160_hd1024_rb2_q159_ema095_pt150_single_sync_attackfreq5'
    for run_dir in (centralized_dir, fedavg_dir, topk_dir, ega_dir):
        run_dir.mkdir()

    _write_json(centralized_dir / 'metrics.json', {'history': [{'round': 0, 'val_mse': 1.2}, {'round': 1, 'val_mse': 1.0}]})
    _write_json(centralized_dir / 'summary.json', {'test': {'mse': 0.8, 'mae': 0.4, 'mape': 8.0}, 'total_parameter_upload_bytes': 0})
    _write_json(fedavg_dir / 'metrics.json', [
        {'round': 0, 'algorithm': 'fedavg', 'val_mse': 1.1, 'val_mae': 0.5, 'val_mape': 10.0, 'total_parameter_upload_bytes': 100},
        {'round': 1, 'algorithm': 'fedavg', 'val_mse': 0.9, 'val_mae': 0.3, 'val_mape': 8.0, 'total_parameter_upload_bytes': 120},
    ])
    fedavg_summary = {'test': {'mse': fedavg_test, 'mae': fedavg_test / 10.0, 'mape': fedavg_test * 10.0}, 'total_parameter_upload_bytes': 220}
    fedavg_summary.update(_attack_payload(*fedavg_attack))
    _write_json(fedavg_dir / 'summary.json', fedavg_summary)
    _write_json(topk_dir / 'metrics.json', [
        {'round': 0, 'algorithm': 'sparse_fedavg', 'active_val_mse': 1.05, 'active_val_mae': 0.48, 'active_val_mape': 9.5, 'total_parameter_upload_bytes': 30},
        {'round': 1, 'algorithm': 'sparse_fedavg', 'active_val_mse': 0.92, 'active_val_mae': 0.31, 'active_val_mape': 8.2, 'total_parameter_upload_bytes': 40},
    ])
    topk_summary = {'test': {'mse': topk_test, 'mae': topk_test / 10.0, 'mape': topk_test * 10.0}, 'total_parameter_upload_bytes': 70}
    topk_summary.update(_attack_payload(*topk_attack))
    _write_json(topk_dir / 'summary.json', topk_summary)
    _write_json(ega_dir / 'metrics.json', [
        {'round': 0, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 1.0, 'protocol_val_mae': 0.45, 'protocol_val_mape': 9.0, 'total_parameter_upload_bytes': 20},
        {'round': 1, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 0.85, 'protocol_val_mae': 0.28, 'protocol_val_mape': 7.5, 'total_parameter_upload_bytes': 25},
    ])
    ega_summary = {'test': {'mse': ega_test, 'mae': ega_test / 10.0, 'mape': ega_test * 10.0}, 'total_parameter_upload_bytes': 45}
    ega_summary.update(_attack_payload(*ega_attack))
    _write_json(ega_dir / 'summary.json', ega_summary)
    return suite_dir


def test_analyze_experiment_suite_discovers_rows_outputs_plots_and_supports_old(tmp_path):
    suite_dir = tmp_path / 'suite' / 'attack_mse'
    suite_dir.mkdir(parents=True)

    centralized_dir = suite_dir / 'centralized_attackfreq5'
    fedavg_dir = suite_dir / 'fedavg_single_sync_attackfreq5'
    topk_dir = suite_dir / 'topk_single_sync_attackfreq5'
    ega_dir = suite_dir / 'ega_ed160_hd1024_rb2_q159_ema095_pt150_single_sync_attackfreq5'
    ega_old_dir = suite_dir / 'ega_ed160_hd1024_rb2_q159_ema095_pt150_single_sync_attackfreq5_old'
    sign_dir = suite_dir / 'sign_single_sync_attackfreq5'
    for run_dir in (centralized_dir, fedavg_dir, topk_dir, ega_dir, ega_old_dir, sign_dir):
        run_dir.mkdir()

    _write_json(centralized_dir / 'metrics.json', {'history': [{'round': 0, 'val_mse': 1.2}, {'round': 1, 'val_mse': 1.0}]})
    _write_json(centralized_dir / 'summary.json', {'test': {'mse': 0.8, 'mae': 0.4}, 'total_parameter_upload_bytes': 0})
    _write_json(fedavg_dir / 'metrics.json', [{'round': 0, 'algorithm': 'fedavg', 'val_mse': 1.1, 'val_mae': 0.5, 'total_parameter_upload_bytes': 100}, {'round': 1, 'algorithm': 'fedavg', 'val_mse': 0.9, 'val_mae': 0.3, 'total_parameter_upload_bytes': 120}])
    fedavg_summary = {'test': {'mse': 0.75, 'mae': 0.075}, 'total_parameter_upload_bytes': 220}
    fedavg_summary.update(_attack_payload(0.81, 0.15, 24))
    _write_json(fedavg_dir / 'summary.json', fedavg_summary)
    _write_json(topk_dir / 'metrics.json', [{'round': 0, 'algorithm': 'sparse_fedavg', 'active_val_mse': 1.05, 'active_val_mae': 0.48, 'total_parameter_upload_bytes': 30}, {'round': 1, 'algorithm': 'sparse_fedavg', 'active_val_mse': 0.92, 'active_val_mae': 0.31, 'total_parameter_upload_bytes': 40}])
    topk_summary = {'test': {'mse': 0.78, 'mae': 0.078}, 'total_parameter_upload_bytes': 70}
    topk_summary.update(_attack_payload(0.95, 0.40, 24))
    _write_json(topk_dir / 'summary.json', topk_summary)
    _write_json(ega_dir / 'metrics.json', [{'round': 0, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 1.0, 'protocol_val_mae': 0.44, 'total_parameter_upload_bytes': 20}, {'round': 1, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 0.85, 'protocol_val_mae': 0.28, 'total_parameter_upload_bytes': 25}])
    ega_summary = {'test': {'mse': 0.74, 'mae': 0.074}, 'total_parameter_upload_bytes': 45}
    ega_summary.update(_attack_payload(1.12, 0.0, 24))
    _write_json(ega_dir / 'summary.json', ega_summary)
    _write_json(ega_old_dir / 'metrics.json', [{'round': 0, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 1.3, 'total_parameter_upload_bytes': 22}, {'round': 1, 'algorithm': 'ega_fedavg', 'protocol_val_mse': 1.0, 'total_parameter_upload_bytes': 27}])
    ega_old_summary = {'test': {'mse': 0.79}, 'total_parameter_upload_bytes': 49}
    ega_old_summary.update(_attack_payload(1.05, 0.1, 24))
    _write_json(ega_old_dir / 'summary.json', ega_old_summary)
    _write_json(sign_dir / 'metrics.json', [{'round': 0, 'algorithm': 'sign_fedavg', 'val_mse': 1.4, 'total_parameter_upload_bytes': 10}])
    _write_json(sign_dir / 'summary.json', {'test': {'mse': 0.9}, 'total_parameter_upload_bytes': 10})

    records = module.discover_runs(suite_dir, 'mse', module.normalize_algorithm_tokens(module.DEFAULT_ALGORITHMS), include_old=False)
    labels = [record.label for record in records]
    assert labels == ['centralized', 'fedavg', 'topk', 'ega_ed160_hd1024_rb2_q159_ema095_pt150']
    assert records[1].val_rounds == [0, 1]
    assert records[1].val_cumulative_upload_bytes == [100, 220]
    assert records[1].attack_present is True
    assert records[0].attack_present is False

    sign_only = module.discover_runs(suite_dir, 'mse', ('sign',), include_old=False)
    assert [record.label for record in sign_only] == ['sign_single_sync_attackfreq5']

    records_with_old = module.discover_runs(suite_dir, 'mse', module.normalize_algorithm_tokens(module.DEFAULT_ALGORITHMS), include_old=True)
    labels_with_old = [record.label for record in records_with_old]
    assert labels_with_old[-1] == 'ega_ed160_hd1024_rb2_q159_ema095_pt150_old'

    rows = module.build_rows(records, metrics=('mse', 'mae'))
    assert rows[1]['label'] == 'fedavg'
    assert rows[1]['fedavg_upload_compression_ratio'] == 1.0
    assert rows[2]['fedavg_upload_compression_ratio'] == 220 / 70
    assert rows[3]['fedavg_upload_compression_ratio'] == 220 / 45
    assert rows[3]['mse_loss_ratio_percent'] == (0.74 - 0.75) / 0.75 * 100.0
    assert rows[3]['mae_relative_percent_vs_fedavg'] == (0.074 - 0.075) / 0.075 * 100.0
    assert rows[1]['attack_primary_metric_name'] == 'budget_recovered_fraction'
    assert rows[1]['attack_overall_avg_primary_metric_value'] == 0.81
    assert rows[1]['attack_success_rate'] == 0.15
    assert rows[1]['attack_evaluations'] == 24
    assert rows[0]['attack_primary_metric_name'] is None

    output_dir = tmp_path / 'analysis'
    csv_path = module.write_summary_csv(rows, output_dir / 'summary.csv', metrics=('mse', 'mae'))
    md_path = module.write_summary_markdown(rows, output_dir / 'summary.md', metrics=('mse', 'mae'))
    round_plot = module.plot_validation_metric_vs_round(records, 'mse', output_dir / 'val_mse_vs_round.png')
    upload_plot = module.plot_validation_metric_vs_upload(records, 'mse', output_dir / 'val_mse_vs_cumulative_upload.png')
    test_plot = module.plot_test_metric_bar(rows, 'mse', output_dir / 'test_mse_bar.png')
    bubble_plot = module.plot_test_metric_vs_upload_bubble(rows, 'mse', output_dir / 'test_mse_vs_upload_bubble.png')

    assert csv_path.exists()
    assert md_path.exists()
    assert round_plot.exists()
    assert upload_plot.exists()
    assert test_plot.exists()
    assert bubble_plot.exists()
    csv_rows = list(csv.DictReader(csv_path.open('r', encoding='utf-8')))
    assert [row['label'] for row in csv_rows] == labels
    assert csv_rows[1]['attack_primary_metric_name'] == 'budget_recovered_fraction'
    assert 'Test MAE' in md_path.read_text(encoding='utf-8')
    assert module.default_output_dir(suite_dir, 'mse').name == 'attack_mse_analysis_mse'


def test_analyze_experiment_suite_aggregates_multiple_seeds_with_std_outputs(tmp_path):
    seed_a = _create_seed_suite(tmp_path, '42', fedavg_test=0.75, topk_test=0.78, ega_test=0.74, fedavg_attack=(0.81, 0.15, 24), topk_attack=(0.95, 0.40, 24), ega_attack=(1.12, 0.0, 24))
    seed_b = _create_seed_suite(tmp_path, '2026', fedavg_test=0.77, topk_test=0.80, ega_test=0.76, fedavg_attack=(0.91, 0.25, 30), topk_attack=(1.05, 0.55, 30), ega_attack=(1.22, 0.05, 30))

    records_by_seed = [
        module.discover_runs(seed_a, 'mse', module.normalize_algorithm_tokens(module.DEFAULT_ALGORITHMS), include_old=False),
        module.discover_runs(seed_b, 'mse', module.normalize_algorithm_tokens(module.DEFAULT_ALGORITHMS), include_old=False),
    ]
    rows = module.aggregate_rows(records_by_seed, metrics=('mse', 'mae'))
    curves = module.aggregate_metric_curves(records_by_seed, 'mse')

    fedavg_row = next(row for row in rows if row['label'] == 'fedavg')
    assert fedavg_row['seed_count'] == 2
    assert fedavg_row['test_mse_mean'] == 0.76
    assert fedavg_row['test_mse_std'] is not None and fedavg_row['test_mse_std'] > 0.0
    assert fedavg_row['attack_primary_metric_name'] == 'budget_recovered_fraction'
    assert fedavg_row['attack_seed_count'] == 2
    assert math.isclose(fedavg_row['attack_overall_avg_primary_metric_value_mean'], 0.86)
    assert fedavg_row['attack_overall_avg_primary_metric_value_std'] is not None and fedavg_row['attack_overall_avg_primary_metric_value_std'] > 0.0
    assert math.isclose(fedavg_row['attack_success_rate_mean'], 0.20)
    assert math.isclose(fedavg_row['attack_evaluations_mean'], 27.0)
    assert math.isclose(fedavg_row['mae_relative_percent_vs_fedavg_mean'], 0.0)

    fedavg_curve = next(curve for curve in curves if curve.label == 'fedavg')
    assert fedavg_curve.rounds == [0, 1]
    assert fedavg_curve.round_mean[0] == 1.1
    assert fedavg_curve.upload_mean[1] == 220.0

    output_dir = tmp_path / 'analysis_multi'
    csv_path = module.write_aggregated_summary_csv(rows, output_dir / 'summary.csv', metrics=('mse', 'mae'))
    md_path = module.write_aggregated_summary_markdown(rows, output_dir / 'summary.md', metrics=('mse', 'mae'))
    round_plot = module.plot_aggregated_validation_metric_vs_round(curves, 'mse', output_dir / 'val_mse_vs_round.png')
    upload_plot = module.plot_aggregated_validation_metric_vs_upload(curves, 'mse', output_dir / 'val_mse_vs_cumulative_upload.png')
    test_plot = module.plot_aggregated_test_metric_bar(rows, 'mse', output_dir / 'test_mse_bar.png')
    bubble_plot = module.plot_aggregated_test_metric_vs_upload_bubble(rows, 'mse', output_dir / 'test_mse_vs_upload_bubble.png')

    assert csv_path.exists()
    assert md_path.exists()
    assert round_plot.exists()
    assert upload_plot.exists()
    assert test_plot.exists()
    assert bubble_plot.exists()
    csv_rows = list(csv.DictReader(csv_path.open('r', encoding='utf-8')))
    assert [row['label'] for row in csv_rows] == ['centralized', 'fedavg', 'topk', 'ega_ed160_hd1024_rb2_q159_ema095_pt150']
    assert csv_rows[1]['test_mse_mean'] == '0.76'
    assert csv_rows[1]['attack_primary_metric_name'] == 'budget_recovered_fraction'
    assert 'Test MAE' in md_path.read_text(encoding='utf-8')
    assert module.default_multi_output_dir([seed_a, seed_b], 'mse').name == 'multiseed_analysis_mse'


def test_analyze_experiment_suite_supports_accuracy_only_runs(tmp_path):
    suite_dir = tmp_path / 'mnist_suite' / 'attack_cross_entropy'
    suite_dir.mkdir(parents=True)
    centralized_dir = suite_dir / 'centralized'
    fedavg_dir = suite_dir / 'fedavg'
    ega_dir = suite_dir / 'ega'
    for run_dir in (centralized_dir, fedavg_dir, ega_dir):
        run_dir.mkdir()

    _write_json(centralized_dir / 'metrics.json', {'history': [{'round': 0, 'val_metrics': {'accuracy': 0.55}}, {'round': 1, 'val_metrics': {'accuracy': 0.60}}]})
    _write_json(centralized_dir / 'summary.json', {'test': {'accuracy': 0.88}, 'total_parameter_upload_bytes': 0})
    _write_json(fedavg_dir / 'metrics.json', [
        {'round': 0, 'primary_metric_name': 'accuracy', 'primary_metric_value': 0.50, 'val_metrics': {'accuracy': 0.50}, 'total_parameter_upload_bytes': 100},
        {'round': 1, 'primary_metric_name': 'accuracy', 'primary_metric_value': 0.66, 'val_metrics': {'accuracy': 0.66}, 'total_parameter_upload_bytes': 120},
    ])
    fedavg_summary = {'test': {'accuracy': 0.90}, 'total_parameter_upload_bytes': 220}
    fedavg_summary.update(_attack_payload(0.2, 0.1, 12))
    _write_json(fedavg_dir / 'summary.json', fedavg_summary)
    _write_json(ega_dir / 'metrics.json', [
        {'round': 0, 'protocol_val_metrics': {'accuracy': 0.48}, 'total_parameter_upload_bytes': 20},
        {'round': 1, 'protocol_val_metrics': {'accuracy': 0.63}, 'total_parameter_upload_bytes': 25},
    ])
    ega_summary = {'test': {'accuracy': 0.91}, 'total_parameter_upload_bytes': 45}
    ega_summary.update(_attack_payload(0.15, 0.0, 12))
    _write_json(ega_dir / 'summary.json', ega_summary)

    records = module.discover_runs(suite_dir, 'mse', module.normalize_algorithm_tokens(('centralized', 'fedavg', 'ega')), include_old=False)
    inferred = module.infer_metrics_from_records(records)
    assert inferred == ('accuracy',)
    rows = module.build_rows(records)
    fedavg_row = next(row for row in rows if row['label'] == 'fedavg')
    assert fedavg_row['test_accuracy'] == 0.90
    assert fedavg_row['accuracy_relative_percent_vs_fedavg'] == 0.0

    output_dir = tmp_path / 'analysis_accuracy'
    csv_path = module.write_summary_csv(rows, output_dir / 'summary.csv')
    md_path = module.write_summary_markdown(rows, output_dir / 'summary.md')
    round_plot = module.plot_validation_metric_vs_round(records, 'accuracy', output_dir / 'val_accuracy_vs_round.png')
    test_plot = module.plot_test_metric_bar(rows, 'accuracy', output_dir / 'test_accuracy_bar.png')
    assert csv_path.exists()
    assert md_path.exists()
    assert round_plot.exists()
    assert test_plot.exists()
    assert 'Test Accuracy' in md_path.read_text(encoding='utf-8')


def test_analyze_experiment_suite_script_exists_and_is_executable():
    assert SCRIPT_PATH.exists()
    assert SCRIPT_PATH.stat().st_mode & 0o111
    content = SCRIPT_PATH.read_text(encoding='utf-8')
    assert content.startswith('#!/usr/bin/env python3')
    assert '--include-old' in content
    assert '--algorithms' in content
    assert '--metrics' in content
    assert 'DEFAULT_ALGORITHMS' in content
    assert 'val_mse_vs_round.png' in content
    assert 'val_mse_vs_cumulative_upload.png' in content
    assert 'test_mse_bar.png' in content
    assert 'test_mse_vs_upload_bubble.png' in content
    assert 'fedavg baseline' in content
    assert 'attack_success_rate' in content
    assert 'attack_overall_avg_primary_metric_value' in content
