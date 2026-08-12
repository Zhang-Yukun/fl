#!/usr/bin/env python3
"""Prepare XT_data wide CSV files into federated client splits."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import pandas as pd


CLIENT_COLUMNS = {"Nd2O3": "氧化钕", "CeO2": "氧化铈", "La2O3": "氧化镧"}
TEST_FILE_PATTERN = re.compile(r"test([1-9])\.csv$")


def reset_output_dir(output_dir: Path) -> None:
    """Remove stale generated split artifacts from a previous preparation run."""

    for child in [output_dir / "clients", output_dir / "server"]:
        if child.exists():
            shutil.rmtree(child)
    for pattern in ("*.csv", "summary.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _test_sort_key(path: Path) -> int:
    """Return the numeric suffix of a ``test{n}.csv`` path."""

    match = TEST_FILE_PATTERN.match(path.name)
    if not match:
        raise ValueError(f"Not a supported test file: {path}")
    return int(match.group(1))


def discover_test_files(input_dir: Path) -> list[Path]:
    """Discover ``test1.csv`` through ``test9.csv`` in numeric order.

    Example:
        ``discover_test_files(Path("XT_data"))`` returns existing test windows.
    """

    files = [path for path in input_dir.glob("test*.csv") if TEST_FILE_PATTERN.match(path.name)]
    return sorted(files, key=_test_sort_key)


def read_xt_wide_csv(path: Path) -> pd.DataFrame:
    """Read one XT_data wide CSV and aggregate duplicate date/project rows.

    Example:
        ``read_xt_wide_csv(Path("new_train.csv"))`` returns columns ``date``
        plus the three Chinese oxide names, sorted by date.
    """

    frame = pd.read_csv(path)
    required = {"date", *CLIENT_COLUMNS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    frame = frame[["date", *CLIENT_COLUMNS.values()]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in CLIENT_COLUMNS.values():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        raise ValueError(f"No valid dated rows in {path}")
    grouped = frame.groupby("date", as_index=False)[list(CLIENT_COLUMNS.values())].mean().sort_values("date")
    grouped[list(CLIENT_COLUMNS.values())] = grouped[list(CLIENT_COLUMNS.values())].interpolate(
        method="linear", limit_direction="both"
    )
    grouped = grouped.ffill().bfill().dropna(subset=list(CLIENT_COLUMNS.values()))
    grouped["date"] = grouped["date"].dt.strftime("%Y-%m-%d")
    return grouped.reset_index(drop=True)


def combine_test_windows(test_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine test windows and re-aggregate overlapping dates by mean."""

    frames = list(test_frames)
    if not frames:
        raise ValueError("At least one test window is required")
    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    grouped = combined.groupby("date", as_index=False)[list(CLIENT_COLUMNS.values())].mean().sort_values("date")
    grouped["date"] = grouped["date"].dt.strftime("%Y-%m-%d")
    return grouped.reset_index(drop=True)


def split_train_val(train_source: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split the XT training source into train and validation."""

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    val_count = max(1, int(round(len(train_source) * val_ratio)))
    if len(train_source) - val_count < 1:
        raise ValueError("Not enough rows to reserve validation data")
    return train_source.iloc[:-val_count].copy(), train_source.iloc[-val_count:].copy()


def _client_frame(wide: pd.DataFrame, client_id: str) -> pd.DataFrame:
    """Extract one client ``date,value`` frame from a wide XT_data frame."""

    column = CLIENT_COLUMNS[client_id]
    return wide[["date", column]].rename(columns={column: "value"})


def _server_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide XT_data frame into long ``date,client,value`` form."""

    rows = []
    for client_id in CLIENT_COLUMNS:
        client = _client_frame(wide, client_id)
        client.insert(1, "client", client_id)
        rows.append(client)
    return pd.concat(rows, ignore_index=True)


def _write_client_split(output_dir: Path, name: str, wide: pd.DataFrame) -> dict[str, int]:
    """Write one split for every client and return per-client row counts."""

    counts = {}
    for client_id in CLIENT_COLUMNS:
        client_dir = output_dir / "clients" / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        frame = _client_frame(wide, client_id)
        frame.to_csv(client_dir / f"{name}.csv", index=False)
        counts[client_id] = len(frame)
    return counts


def prepare_xt_data(input_dir: Path, output_dir: Path, val_ratio: float = 0.1) -> dict[str, object]:
    """Prepare XT_data CSV files into the framework's federated split layout.

    Example:
        ``prepare_xt_data(Path("XT_data"), Path("data/rare_earth_rawdata2"))``
        writes client train/val/test CSVs and the individual test windows.
    """

    train_path = input_dir / "new_train.csv"
    if not train_path.exists():
        train_path = input_dir / "merge_large.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Cannot find new_train.csv or merge_large.csv in {input_dir}")
    test_paths = discover_test_files(input_dir)
    if not test_paths:
        fallback = input_dir / "new_test.csv"
        if not fallback.exists():
            raise FileNotFoundError(f"Cannot find test1..test9.csv or new_test.csv in {input_dir}")
        test_paths = [fallback]

    output_dir.mkdir(parents=True, exist_ok=True)
    reset_output_dir(output_dir)
    (output_dir / "clients").mkdir(parents=True, exist_ok=True)
    (output_dir / "server").mkdir(parents=True, exist_ok=True)

    train_source = read_xt_wide_csv(train_path)
    train_frame, val_frame = split_train_val(train_source, val_ratio)
    test_windows = {path.name: read_xt_wide_csv(path) for path in test_paths}
    test_frame = combine_test_windows(test_windows.values())

    split_counts = {
        "train": _write_client_split(output_dir, "train", train_frame),
        "val": _write_client_split(output_dir, "val", val_frame),
        "test": _write_client_split(output_dir, "test", test_frame),
    }
    for test_name, frame in test_windows.items():
        _write_client_split(output_dir, test_name.removesuffix(".csv"), frame)

    server_dir = output_dir / "server"
    for split_name, frame in {"train": train_frame, "val": val_frame, "test": test_frame}.items():
        _server_frame(frame).to_csv(server_dir / f"{split_name}.csv", index=False)
    for test_name, frame in test_windows.items():
        _server_frame(frame).to_csv(server_dir / test_name, index=False)

    train_source.to_csv(output_dir / "merged_wide.csv", index=False)
    test_frame.to_csv(output_dir / "test_wide.csv", index=False)
    for test_name, frame in test_windows.items():
        frame.to_csv(output_dir / test_name, index=False)

    summary = {
        "source": "XT_data",
        "input_dir": str(input_dir),
        "train_source_file": train_path.name,
        "val_ratio": val_ratio,
        "train_source_rows": len(train_source),
        "train_rows": len(train_frame),
        "val_rows": len(val_frame),
        "test_rows": len(test_frame),
        "split_counts": split_counts,
        "test_windows": {
            name: {
                "rows": len(frame),
                "start": str(frame["date"].iloc[0]),
                "end": str(frame["date"].iloc[-1]),
            }
            for name, frame in test_windows.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Parse CLI arguments and prepare XT_data files."""

    parser = argparse.ArgumentParser(description="Prepare XT_data CSV files for federated training")
    parser.add_argument("--input-dir", default="../Time-Series-Prediction/dataset/XT_data")
    parser.add_argument("--output-dir", default="../data/rare_earth_rawdata2")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    summary = prepare_xt_data(Path(args.input_dir), Path(args.output_dir), args.val_ratio)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
