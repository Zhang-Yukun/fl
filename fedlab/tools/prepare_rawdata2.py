#!/usr/bin/env python3
"""Prepare rawdata2 rare-earth Excel files into federated train/val/test splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


CLIENT_NAME_BY_CHINESE = {"氧化钕": "Nd2O3", "氧化铈": "CeO2", "氧化镧": "La2O3"}


def reset_output_dir(output_dir: Path) -> None:
    """Remove stale generated artifacts before rebuilding the rawdata2 dataset."""

    for child in [output_dir / "clients", output_dir / "server"]:
        if child.exists():
            shutil.rmtree(child)
    for pattern in ("*.csv", "*.json"):
        for artifact in output_dir.glob(pattern):
            if artifact.is_file():
                artifact.unlink()



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


def _build_interpolated_merged_frame(series_by_client: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a daily wide frame with linear interpolation across missing dates.

    Example:
        ``_build_interpolated_merged_frame({"Nd2O3": daily_frame})`` returns a
        daily wide table spanning the full min-max date range.
    """

    if not series_by_client:
        raise ValueError("At least one client series is required")
    merged = []
    for client_id, daily in series_by_client.items():
        wide = daily.copy()
        wide["date"] = pd.to_datetime(wide["date"], errors="coerce")
        wide = wide.dropna(subset=["date"]).rename(columns={"value": client_id}).set_index("date")
        merged.append(wide[[client_id]])
    wide_frame = pd.concat(merged, axis=1, sort=True).sort_index()
    full_index = pd.date_range(wide_frame.index.min(), wide_frame.index.max(), freq="D")
    wide_frame = wide_frame.reindex(full_index)
    wide_frame = wide_frame.interpolate(method="linear", limit_direction="both").ffill().bfill()
    wide_frame.index.name = "date"
    return wide_frame.reset_index().assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))


def _split_merged_frame_by_client(
    merged_frame: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Split an interpolated wide frame into per-client chronological partitions."""

    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for client_id in [column for column in merged_frame.columns if column != "date"]:
        client_frame = merged_frame[["date", client_id]].rename(columns={client_id: "value"})
        splits[client_id] = _split_frame(client_frame, train_ratio=train_ratio, val_ratio=val_ratio)
    return splits


def prepare_rawdata2(raw_dir: Path, output_dir: Path) -> dict[str, dict[str, int]]:
    """Prepare all rawdata2 Excel files into client and server CSV files.

    Example:
        ``prepare_rawdata2(Path("rawdata2"), Path("data/rare_earth_rawdata2"))``.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    reset_output_dir(output_dir)
    (output_dir / "clients").mkdir(parents=True, exist_ok=True)
    (output_dir / "server").mkdir(parents=True, exist_ok=True)
    summary = {}
    server_train = []
    server_val = []
    server_test = []
    series_by_client: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*.xls")):
        client_id, daily = _read_daily_series(path)
        series_by_client[client_id] = daily

    merged_frame = _build_interpolated_merged_frame(series_by_client)
    split_by_client = _split_merged_frame_by_client(merged_frame)
    for client_id, (train, val, test) in split_by_client.items():
        client_dir = output_dir / "clients" / client_id
        client_dir.mkdir(parents=True, exist_ok=True)
        for name, split in {"train": train, "val": val, "test": test}.items():
            split.to_csv(client_dir / f"{name}.csv", index=False)
        for split_name, split in {"train": train, "val": val, "test": test}.items():
            if split_name == "train":
                target = server_train
            elif split_name == "val":
                target = server_val
            else:
                target = server_test
            combined = split.copy()
            combined.insert(1, "client", client_id)
            target.append(combined)
        summary[client_id] = {"total": len(train) + len(val) + len(test), "train": len(train), "val": len(val), "test": len(test)}
    pd.concat(server_train, ignore_index=True).to_csv(output_dir / "server" / "train.csv", index=False)
    pd.concat(server_val, ignore_index=True).to_csv(output_dir / "server" / "val.csv", index=False)
    pd.concat(server_test, ignore_index=True).to_csv(output_dir / "server" / "test.csv", index=False)
    merged_frame.to_csv(output_dir / "merged_wide.csv", index=False)
    summary_payload = {
        "source": "raw_data",
        "input_dir": str(raw_dir),
        "split_strategy": "chronological_8_1_1",
        "rows": summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_payload


def main() -> None:
    """Parse CLI arguments and prepare rawdata2 files."""

    parser = argparse.ArgumentParser(description="Prepare rawdata2 rare-earth data for federated training")
    parser.add_argument("--raw-dir", default="../data/raw_data")
    parser.add_argument("--output-dir", default="../data/rare_earth_rawdata2")
    args = parser.parse_args()
    summary = prepare_rawdata2(Path(args.raw_dir), Path(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
