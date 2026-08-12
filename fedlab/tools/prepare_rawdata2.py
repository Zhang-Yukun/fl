#!/usr/bin/env python3
"""Prepare rawdata2 rare-earth Excel files into federated train/val/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLIENT_NAME_BY_CHINESE = {"氧化钕": "Nd2O3", "氧化铈": "CeO2", "氧化镧": "La2O3"}


def _read_daily_series(path: Path) -> tuple[str, pd.DataFrame]:
    """Read one CBC Excel file and return a daily RMB average price series."""

    raw = pd.read_excel(path, header=None)
    if raw.empty:
        raise ValueError(f"Empty Excel file: {path}")
    header_row = raw.index[raw.iloc[:, 0].astype(str).eq("日期")]
    if len(header_row) == 0:
        raise ValueError(f"Cannot locate header row with 日期 in {path}")
    header_idx = int(header_row[0])
    data = raw.iloc[header_idx + 1 :].copy()
    data.columns = raw.iloc[header_idx].tolist()
    required = ["日期", "品名", "平均价", "单位", "价格类型", "付款方式"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    data = data[required].copy()
    data["date"] = pd.to_datetime(data["日期"], errors="coerce")
    data["value"] = pd.to_numeric(data["平均价"], errors="coerce")
    mask = (
        data["date"].notna()
        & data["value"].notna()
        & data["单位"].astype(str).str.contains("元/吨", regex=False)
        & data["价格类型"].astype(str).str.contains("出厂价", regex=False)
        & data["付款方式"].astype(str).str.contains("含税现款", regex=False)
    )
    data = data.loc[mask].copy()
    if data.empty:
        raise ValueError(f"No RMB ex-works tax-included rows found in {path}")
    oxide_cn = str(data["品名"].mode().iloc[0])
    client_id = CLIENT_NAME_BY_CHINESE.get(oxide_cn, oxide_cn)
    daily = data.groupby("date", as_index=False)["value"].mean().sort_values("date")
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    return client_id, daily


def _split_frame(frame: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split a time-ordered frame into chronological train/val/test partitions."""

    train_end = int(len(frame) * train_ratio)
    val_end = train_end + int(len(frame) * val_ratio)
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:val_end].copy(), frame.iloc[val_end:].copy()


def prepare_rawdata2(raw_dir: Path, output_dir: Path) -> dict[str, dict[str, int]]:
    """Prepare all rawdata2 Excel files into client and server CSV files.

    Example:
        ``prepare_rawdata2(Path("rawdata2"), Path("data/rare_earth_rawdata2"))``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clients").mkdir(parents=True, exist_ok=True)
    (output_dir / "server").mkdir(parents=True, exist_ok=True)
    summary = {}
    server_val = []
    server_test = []
    merged = []
    for path in sorted(raw_dir.glob("*.xls")):
        client_id, daily = _read_daily_series(path)
        train, val, test = _split_frame(daily)
        client_dir = output_dir / "clients" / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        for name, split in {"train": train, "val": val, "test": test}.items():
            split.to_csv(client_dir / f"{name}.csv", index=False)
        for split_name, split, target in [("val", val, server_val), ("test", test, server_test)]:
            combined = split.copy()
            combined.insert(1, "client", client_id)
            target.append(combined)
        wide = daily.rename(columns={"value": client_id}).set_index("date")
        merged.append(wide)
        summary[client_id] = {"total": len(daily), "train": len(train), "val": len(val), "test": len(test)}
    pd.concat(server_val, ignore_index=True).to_csv(output_dir / "server" / "val.csv", index=False)
    pd.concat(server_test, ignore_index=True).to_csv(output_dir / "server" / "test.csv", index=False)
    pd.concat(merged, axis=1).sort_index().reset_index().to_csv(output_dir / "merged_wide.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """Parse CLI arguments and prepare rawdata2 files."""

    parser = argparse.ArgumentParser(description="Prepare rawdata2 rare-earth data for federated training")
    parser.add_argument("--raw-dir", default="../Time-Series-Prediction/dataset/data_preprocess/rawdata2")
    parser.add_argument("--output-dir", default="../data/rare_earth_rawdata2")
    args = parser.parse_args()
    summary = prepare_rawdata2(Path(args.raw_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
