import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[2] / 'fedlab' / 'tools' / 'plot_dataset_splits.py'
spec = importlib.util.spec_from_file_location('plot_dataset_splits', SCRIPT_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_plot_dataset_splits_loads_and_plots(tmp_path):
    """The split plotting helper loads split CSVs, indexes per client, and writes plots."""

    data_dir = tmp_path / 'dataset'
    data_dir.mkdir()
    pd.DataFrame([
        {'date': '2024-01-01', 'client': 'A', 'value': 1.0},
        {'date': '2024-01-02', 'client': 'A', 'value': 2.0},
        {'date': '2024-01-01', 'client': 'B', 'value': 10.0},
    ]).to_csv(data_dir / 'train.csv', index=False)
    pd.DataFrame([
        {'date': '2024-01-03', 'client': 'A', 'value': 3.0},
        {'date': '2024-01-02', 'client': 'B', 'value': 11.0},
    ]).to_csv(data_dir / 'val.csv', index=False)
    pd.DataFrame([
        {'date': '2024-01-04', 'client': 'A', 'value': 4.0},
        {'date': '2024-01-03', 'client': 'B', 'value': 12.0},
    ]).to_csv(data_dir / 'test.csv', index=False)

    frame = module.load_all_splits(data_dir)
    client_a = frame[frame['client'] == 'A']
    client_b = frame[frame['client'] == 'B']
    assert list(client_a['sequence_index']) == [0, 1, 2, 3]
    assert list(client_b['sequence_index']) == [0, 1, 2]
    assert set(frame['split']) == {'train', 'val', 'test'}

    output = tmp_path / 'plot.png'
    module.plot_split_series(frame, output, 'demo')
    assert output.exists()
    assert output.stat().st_size > 0

    split_outputs = module.plot_splits_separately(frame, output, 'demo separate')
    assert len(split_outputs) == 3
    for split_output in split_outputs:
        assert split_output.exists()
        assert split_output.stat().st_size > 0


def test_plot_dataset_splits_script_exists_and_is_executable():
    """The split plotting helper is exposed as an executable script."""

    assert SCRIPT_PATH.exists()
    assert SCRIPT_PATH.stat().st_mode & 0o111
    content = SCRIPT_PATH.read_text(encoding='utf-8')
    assert content.startswith('#!/usr/bin/env python3')
    assert '--separate-splits' in content


def test_plot_dataset_splits_ignores_auxiliary_test_windows(tmp_path):
    """The split plotting helper should ignore test1.csv style auxiliary files."""

    data_dir = tmp_path / 'dataset'
    data_dir.mkdir()
    pd.DataFrame([{'date': '2024-01-01', 'client': 'A', 'value': 1.0}]).to_csv(data_dir / 'train.csv', index=False)
    pd.DataFrame([{'date': '2024-01-02', 'client': 'A', 'value': 2.0}]).to_csv(data_dir / 'val.csv', index=False)
    pd.DataFrame([{'date': '2024-01-03', 'client': 'A', 'value': 3.0}]).to_csv(data_dir / 'test.csv', index=False)
    pd.DataFrame([{'date': '2025-01-01', 'client': 'A', 'value': 999.0}]).to_csv(data_dir / 'test1.csv', index=False)

    frame = module.load_all_splits(data_dir)

    assert len(frame[frame['split'] == 'test']) == 1
    assert float(frame[frame['split'] == 'test']['value'].iloc[0]) == 3.0
