import importlib.util
from pathlib import Path

import pandas as pd


def _load_prepare_xt_data():
    module_path = Path(__file__).parents[2] / "fedlab" / "tools" / "prepare_xt_data.py"
    spec = importlib.util.spec_from_file_location("prepare_xt_data_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.prepare_xt_data


prepare_xt_data = _load_prepare_xt_data()


def _write_wide(path, rows):
    frame = pd.DataFrame(rows, columns=["date", "氧化钕", "氧化铈", "氧化镧"])
    frame.to_csv(path, index=False)


def test_prepare_xt_data_aggregates_and_keeps_test_windows(tmp_path):
    """XT_data preprocessing writes client splits and preserves each test window."""

    input_dir = tmp_path / "XT_data"
    output_dir = tmp_path / "rare_earth_rawdata2"
    input_dir.mkdir()
    _write_wide(
        input_dir / "new_train.csv",
        [
            ("2020-01-01", 1.0, 10.0, 100.0),
            ("2020-01-01", 3.0, 30.0, 300.0),
            ("2020-01-02", 4.0, 40.0, 400.0),
            ("2020-01-03", 5.0, 50.0, 500.0),
            ("2020-01-04", 6.0, 60.0, 600.0),
        ],
    )
    _write_wide(input_dir / "test1.csv", [("2020-02-01", 7.0, 70.0, 700.0), ("2020-02-02", 8.0, 80.0, 800.0)])
    _write_wide(input_dir / "test2.csv", [("2020-02-02", 10.0, 100.0, 1000.0), ("2020-02-03", 12.0, 120.0, 1200.0)])

    summary = prepare_xt_data(input_dir, output_dir, val_ratio=0.5)

    nd_train = pd.read_csv(output_dir / "clients" / "Nd2O3" / "train.csv")
    nd_val = pd.read_csv(output_dir / "clients" / "Nd2O3" / "val.csv")
    nd_test = pd.read_csv(output_dir / "clients" / "Nd2O3" / "test.csv")
    nd_test1 = pd.read_csv(output_dir / "clients" / "Nd2O3" / "test1.csv")

    assert summary["source"] == "XT_data"
    assert nd_train["date"].tolist() == ["2020-01-01", "2020-01-02"]
    assert nd_train["value"].tolist()[0] == 2.0
    assert nd_val["date"].tolist() == ["2020-01-03", "2020-01-04"]
    assert nd_test["date"].tolist() == ["2020-02-01", "2020-02-02", "2020-02-03"]
    assert nd_test["value"].tolist()[1] == 9.0
    assert nd_test1["date"].tolist() == ["2020-02-01", "2020-02-02"]
    assert (output_dir / "server" / "test1.csv").exists()
    assert (output_dir / "test_wide.csv").exists()
