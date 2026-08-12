#!/usr/bin/env python3
"""Plot train/validation/test series from one dataset directory.

Example:
    python -m fedlab.tools.plot_dataset_splits         ../data/rare_earth_rawdata2/server         --output ../data/rare_earth_rawdata2/server/split_series.png

The script searches the target directory for ``train*.csv``, ``val*.csv``, and
``test*.csv`` files. By default it renders one long figure with per-client
subplots and colored train/validation/test segments. With ``--separate-splits``
it writes three figures instead, one for each split, still grouped by client.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

SPLIT_PATTERNS = {
    'train': 'train*.csv',
    'val': 'val*.csv',
    'test': 'test*.csv',
}
SPLITS = ('train', 'val', 'test')
SPLIT_COLORS = {
    'train': '#1f77b4',
    'val': '#ff7f0e',
    'test': '#2ca02c',
}
FIGURE_WIDTH = 18
CLIENT_PANEL_HEIGHT = 3.8


def find_split_files(data_dir: Path, split: str) -> list[Path]:
    """Return all CSV files for one split under the target directory."""

    pattern = SPLIT_PATTERNS[split]
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())



def load_split_frame(paths: Iterable[Path], split: str) -> pd.DataFrame:
    """Load and concatenate one split into a single frame."""

    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        if 'value' not in frame.columns:
            raise ValueError(f'{path} is missing required column: value')
        frame = frame.copy()
        frame['split'] = split
        frame['source_file'] = path.name
        if 'client' not in frame.columns:
            frame['client'] = 'default'
        if 'date' in frame.columns:
            frame['date'] = pd.to_datetime(frame['date'])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=['value', 'split', 'source_file', 'client'])
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ('client', 'date', 'source_file') if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, kind='stable').reset_index(drop=True)
    return merged



def _assign_client_indices(frame: pd.DataFrame) -> pd.DataFrame:
    """Assign per-client sequence indices for plotting."""

    frames_with_index: list[pd.DataFrame] = []
    for client, client_frame in frame.groupby('client', sort=True):
        client_frame = client_frame.reset_index(drop=True).copy()
        client_frame['sequence_index'] = range(len(client_frame))
        frames_with_index.append(client_frame)
    return pd.concat(frames_with_index, ignore_index=True) if frames_with_index else frame.copy()



def load_all_splits(data_dir: Path) -> pd.DataFrame:
    """Load train, validation, and test CSVs from one directory."""

    parts: list[pd.DataFrame] = []
    for split in SPLITS:
        frame = load_split_frame(find_split_files(data_dir, split), split)
        if not frame.empty:
            parts.append(frame)
    if not parts:
        raise FileNotFoundError(f'No train/val/test CSV files found in {data_dir}')
    combined = pd.concat(parts, ignore_index=True)
    return _assign_client_indices(combined)



def _prepare_axes(clients: list[str]):
    """Create one subplot row per client."""

    fig, axes = plt.subplots(
        len(clients),
        1,
        figsize=(FIGURE_WIDTH, max(CLIENT_PANEL_HEIGHT * len(clients), CLIENT_PANEL_HEIGHT)),
        sharex=False,
        squeeze=False,
    )
    return fig, axes[:, 0]



def plot_split_series(frame: pd.DataFrame, output_path: Path, title: str | None = None) -> Path:
    """Plot concatenated split segments into per-client subplots."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clients = list(frame['client'].dropna().astype(str).unique())
    if not clients:
        raise ValueError('No client data available for plotting')

    fig, axes = _prepare_axes(clients)

    for axis, client in zip(axes, clients):
        client_frame = frame[frame['client'].astype(str) == client].copy()
        cursor = 0
        for split in SPLITS:
            segment = client_frame[client_frame['split'] == split]
            if segment.empty:
                continue
            x = segment['sequence_index'].to_numpy()
            y = segment['value'].to_numpy()
            axis.plot(x, y, color=SPLIT_COLORS[split], linewidth=1.2, label=f'{split} ({len(segment)})')
            if cursor > 0:
                axis.axvline(x=cursor - 0.5, color='#888888', linestyle='--', linewidth=0.8, alpha=0.8)
            cursor += len(segment)
        axis.set_ylabel('Value')
        axis.set_title(f'Client: {client}')
        axis.grid(True, alpha=0.25)
        axis.legend(loc='upper right')

    axes[-1].set_xlabel('Sequence index')
    fig.suptitle(title or f'Dataset splits by client: {output_path.stem}', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path



def plot_splits_separately(frame: pd.DataFrame, output_base: Path, title: str | None = None) -> list[Path]:
    """Plot one figure per split, with per-client subplots.

    Example:
        ``plot_splits_separately(frame, Path('split_series.png'))`` writes
        ``split_series_train.png``, ``split_series_val.png``, and ``split_series_test.png``.
    """

    output_base.parent.mkdir(parents=True, exist_ok=True)
    clients = list(frame['client'].dropna().astype(str).unique())
    if not clients:
        raise ValueError('No client data available for plotting')

    outputs: list[Path] = []
    for split in SPLITS:
        split_frame = frame[frame['split'] == split].copy()
        if split_frame.empty:
            continue
        split_frame = _assign_client_indices(split_frame)
        fig, axes = _prepare_axes(clients)
        for axis, client in zip(axes, clients):
            client_frame = split_frame[split_frame['client'].astype(str) == client]
            if not client_frame.empty:
                x = client_frame['sequence_index'].to_numpy()
                y = client_frame['value'].to_numpy()
                axis.plot(x, y, color=SPLIT_COLORS[split], linewidth=1.2, label=f'{split} ({len(client_frame)})')
            axis.set_ylabel('Value')
            axis.set_title(f'Client: {client}')
            axis.grid(True, alpha=0.25)
            axis.legend(loc='upper right')
        axes[-1].set_xlabel('Sequence index')
        split_output = output_base.with_name(f'{output_base.stem}_{split}{output_base.suffix}')
        fig.suptitle(title or f'{split} by client: {output_base.stem}', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(split_output, dpi=180)
        plt.close(fig)
        outputs.append(split_output)
    return outputs



def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser for the split plotting helper."""

    parser = argparse.ArgumentParser(description='Plot train/val/test CSV series from one directory.')
    parser.add_argument('data_dir', type=Path, help='Directory containing train*.csv, val*.csv, and test*.csv files')
    parser.add_argument('--output', type=Path, default=None, help='Output image path. Defaults to <data_dir>/split_series.png')
    parser.add_argument('--title', type=str, default=None, help='Optional chart title')
    parser.add_argument('--separate-splits', action='store_true', help='Write one figure per split instead of one combined figure')
    return parser



def main() -> None:
    """Run the plotting CLI."""

    args = build_argparser().parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(f'{data_dir} is not a directory')
    output = args.output.resolve() if args.output is not None else data_dir / 'split_series.png'
    frame = load_all_splits(data_dir)
    counts = frame['split'].value_counts().to_dict()
    clients = sorted(frame['client'].astype(str).unique())

    if args.separate_splits:
        outputs = plot_splits_separately(frame, output, args.title)
        print(f'saved_plots: {[str(path) for path in outputs]}')
    else:
        plot_split_series(frame, output, args.title)
        print(f'saved_plot: {output}')
    print(f'clients: {clients}')
    print(f'split_counts: train={counts.get("train", 0)} val={counts.get("val", 0)} test={counts.get("test", 0)}')


if __name__ == '__main__':
    main()
