"""Replay one saved model on the configured test split."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from fedlab.datasets import build_federated_loaders
from fedlab.engine.training import evaluate
from fedlab.federated.algorithms import configure_random_seed, configure_torch_runtime, resolve_device
from fedlab.modeling import build_model
from fedlab.tools.replay_saved_update_common import default_config_path
from fedlab.utils.artifacts import save_experiment_config
from fedlab.utils.config import load_config
from fedlab.utils.logging import setup_logging


def build_evaluation_config(config_path: Path, overrides: list[str], data_dir: Path | None) -> dict[str, Any]:
    """Load one evaluation config and optionally override the dataset split directory."""

    final_overrides = list(overrides)
    if data_dir is not None:
        final_overrides.append(f"data.split_dir={data_dir}")
    config = load_config(config_path, final_overrides)
    replay_config = copy.deepcopy(config)
    replay_config.setdefault('tracking', {})['enabled'] = False
    return replay_config


def evaluate_saved_model(
    model_path: Path,
    *,
    config_path: Path,
    output_dir: Path,
    overrides: list[str],
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one serialized model against the configured shared test loader."""

    config = build_evaluation_config(config_path, overrides, data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, config.get('runtime', {}).get('log_level', 'INFO'))
    save_experiment_config(config, output_dir, config.get('artifacts', {}).get('config_formats'))
    configure_torch_runtime(config)
    configure_random_seed(config)
    device = resolve_device(config)
    _, _, test_loader = build_federated_loaders(config)
    model = build_model(config).to(device)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    start = time.perf_counter()
    metrics = evaluate(model, test_loader, device)
    elapsed = time.perf_counter() - start
    mode = str(config.get('experiment', {}).get('mode', 'federated')).lower()
    summary = {
        'test': metrics,
        'protocol_test': metrics if mode == 'federated' else metrics,
        'evaluation_mode': 'offline_saved_model_test',
        'mode': mode,
        'task_type': str(config.get('task', {}).get('type', 'forecasting')).lower(),
        'source_model_path': str(model_path),
        'source_config_path': str(config_path),
        'source_data_split_dir': str(Path(config.get('data', {}).get('split_dir', '')).expanduser()),
        'elapsed_time_seconds': elapsed,
    }
    with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (output_dir / 'test_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    logger.info('Offline saved-model evaluation finished in {:.4f}s with metrics {}', elapsed, metrics)
    return {
        'model_path': str(model_path),
        'config_path': str(config_path),
        'output_dir': str(output_dir),
        'test_metrics': metrics,
        'test_metrics_path': str(output_dir / 'test_metrics.json'),
        'test_summary_path': str(output_dir / 'test_summary.json'),
    }


def main() -> None:
    """Run the saved-model evaluation CLI."""

    parser = argparse.ArgumentParser(description='Evaluate one saved model on the configured shared test split')
    parser.add_argument('model_path', type=Path, help='Path to one saved model.pt state dict')
    parser.add_argument('--config', type=Path, default=None, help='Optional config artifact path; defaults to model_path parent config.*')
    parser.add_argument('--data-dir', type=Path, default=None, help='Optional split directory overriding data.split_dir')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory receiving detached test artifacts')
    parser.add_argument('--override', action='append', default=[], help='Optional config overrides applied before testing')
    args = parser.parse_args()

    model_path = args.model_path.expanduser().resolve()
    config_path = args.config.expanduser().resolve() if args.config is not None else default_config_path(model_path.parent)
    data_dir = args.data_dir.expanduser().resolve() if args.data_dir is not None else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else (model_path.parent / 'offline_test_replay')
    payload = evaluate_saved_model(
        model_path,
        config_path=config_path,
        output_dir=output_dir,
        overrides=list(args.override),
        data_dir=data_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
