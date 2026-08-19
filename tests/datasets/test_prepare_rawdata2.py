import importlib.util
from pathlib import Path

import pandas as pd


def _load_prepare_rawdata2_module():
    module_path = Path(__file__).parents[2] / "fedlab" / "tools" / "prepare_rawdata2.py"
    spec = importlib.util.spec_from_file_location("prepare_rawdata2_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_rawdata2_module = _load_prepare_rawdata2_module()
_build_interpolated_merged_frame = prepare_rawdata2_module._build_interpolated_merged_frame
_split_merged_frame_by_client = prepare_rawdata2_module._split_merged_frame_by_client
reset_output_dir = prepare_rawdata2_module.reset_output_dir


def test_build_interpolated_merged_frame_fills_missing_days_and_values():
    """rawdata2 merged_wide should be daily and linearly interpolated."""

    series_by_client = {
        "Nd2O3": pd.DataFrame({
            "date": ["2020-01-01", "2020-01-03"],
            "value": [1.0, 3.0],
        }),
        "CeO2": pd.DataFrame({
            "date": ["2020-01-02", "2020-01-04"],
            "value": [10.0, 14.0],
        }),
    }

    merged = _build_interpolated_merged_frame(series_by_client)

    assert merged["date"].tolist() == [
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
    ]
    assert merged["Nd2O3"].tolist() == [1.0, 2.0, 3.0, 3.0]
    assert merged["CeO2"].tolist() == [10.0, 10.0, 12.0, 14.0]
    assert not merged.isna().any().any()


def test_split_merged_frame_by_client_uses_interpolated_dense_series():
    """Client/server splits should come from the interpolated daily frame."""

    merged = pd.DataFrame({
        "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
        "Nd2O3": [1.0, 2.0, 3.0, 4.0],
        "CeO2": [10.0, 11.0, 12.0, 13.0],
    })

    split_by_client = _split_merged_frame_by_client(merged, train_ratio=0.5, val_ratio=0.25)
    nd_train, nd_val, nd_test = split_by_client["Nd2O3"]
    ce_train, ce_val, ce_test = split_by_client["CeO2"]

    assert nd_train["date"].tolist() == ["2020-01-01", "2020-01-02"]
    assert nd_val["date"].tolist() == ["2020-01-03"]
    assert nd_test["date"].tolist() == ["2020-01-04"]
    assert ce_train["value"].tolist() == [10.0, 11.0]
    assert ce_val["value"].tolist() == [12.0]
    assert ce_test["value"].tolist() == [13.0]


def test_reset_output_dir_removes_stale_generated_artifacts(tmp_path):
    """Rebuilding rawdata2 output should remove stale split artifacts first."""

    clients_dir = tmp_path / "clients" / "Nd2O3"
    server_dir = tmp_path / "server"
    clients_dir.mkdir(parents=True)
    server_dir.mkdir(parents=True)
    (clients_dir / "test1.csv").write_text("stale", encoding="utf-8")
    (server_dir / "test9.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "test_wide.csv").write_text("stale", encoding="utf-8")
    (tmp_path / "summary.json").write_text("{}", encoding="utf-8")

    reset_output_dir(tmp_path)

    assert not (tmp_path / "clients").exists()
    assert not (tmp_path / "server").exists()
    assert not (tmp_path / "test_wide.csv").exists()
    assert not (tmp_path / "summary.json").exists()
