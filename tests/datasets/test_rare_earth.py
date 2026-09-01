import numpy as np
import pytest

from fedlab.datasets.rare_earth import Standardizer, WindowDataset, split_array


def test_standardizer_roundtrip_and_split_array():
    values = np.arange(20, dtype="float32")[:, None]
    train, val, test = split_array(values, 0.5, 0.25)
    assert len(train) == 10
    assert len(val) == 5
    assert len(test) == 5
    scaler = Standardizer.fit(train)
    restored = scaler.inverse_transform(scaler.transform(train))
    np.testing.assert_allclose(restored, train, rtol=1e-6, atol=1e-6)


def test_window_dataset_returns_input_and_target_windows():
    values = np.arange(20, dtype="float32")[:, None]
    dataset = WindowDataset(values, seq_len=4, pred_len=2)
    x, y = dataset[0]
    assert x.shape == (4, 1)
    assert y.shape == (2, 1)


def test_build_federated_loaders_from_split_dir(tmp_path):
    from fedlab.datasets.rare_earth import build_federated_loaders

    for client in ["Nd2O3", "CeO2", "La2O3"]:
        client_dir = tmp_path / "clients" / client
        client_dir.mkdir(parents=True)
        for split, start, length in [("train", 0, 40), ("val", 40, 30), ("test", 70, 30)]:
            dates = [f"2020-01-{(idx % 28) + 1:02d}" for idx in range(start, start + length)]
            values = np.arange(start, start + length, dtype="float32")
            rows = "date,value\n" + "\n".join(f"{date},{value}" for date, value in zip(dates, values)) + "\n"
            (client_dir / f"{split}.csv").write_text(rows, encoding="utf-8")
    config = {"data": {"split_dir": str(tmp_path), "clients": ["Nd2O3", "CeO2", "La2O3"], "seq_len": 4, "pred_len": 2, "batch_size": 8}}
    train_loaders, val_loader, test_loader = build_federated_loaders(config)
    assert set(train_loaders) == {"Nd2O3", "CeO2", "La2O3"}
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert getattr(train_loaders["Nd2O3"], "scaler", None) is not None
    assert getattr(val_loader, "scaler", None) is not None
    assert getattr(test_loader, "scaler", None) is not None


def test_build_federated_loaders_respects_runtime_seed_for_train_shuffle(tmp_path):
    from fedlab.datasets.rare_earth import build_federated_loaders

    for client in ["Nd2O3", "CeO2", "La2O3"]:
        client_dir = tmp_path / "clients" / client
        client_dir.mkdir(parents=True)
        for split, start, length in [("train", 0, 50), ("val", 50, 20), ("test", 70, 20)]:
            dates = [f"2020-01-{(idx % 28) + 1:02d}" for idx in range(start, start + length)]
            values = np.arange(start, start + length, dtype="float32")
            rows = "date,value\n" + "\n".join(f"{date},{value}" for date, value in zip(dates, values)) + "\n"
            (client_dir / f"{split}.csv").write_text(rows, encoding="utf-8")

    def first_batch(seed: int):
        config = {
            "runtime": {"seed": seed},
            "data": {
                "split_dir": str(tmp_path),
                "clients": ["Nd2O3", "CeO2", "La2O3"],
                "seq_len": 4,
                "pred_len": 2,
                "batch_size": 8,
                "shuffle_train": True,
            },
        }
        train_loaders, _, _ = build_federated_loaders(config)
        x, y = next(iter(train_loaders["Nd2O3"]))
        return x.numpy().copy(), y.numpy().copy()

    first_x_a, first_y_a = first_batch(2026)
    first_x_b, first_y_b = first_batch(2026)
    first_x_c, first_y_c = first_batch(2027)

    np.testing.assert_allclose(first_x_a, first_x_b)
    np.testing.assert_allclose(first_y_a, first_y_b)
    assert not (np.allclose(first_x_a, first_x_c) and np.allclose(first_y_a, first_y_c))


def test_build_federated_loaders_can_disable_train_shuffle(tmp_path):
    from fedlab.datasets.rare_earth import build_federated_loaders

    for client in ["Nd2O3", "CeO2", "La2O3"]:
        client_dir = tmp_path / "clients" / client
        client_dir.mkdir(parents=True)
        for split, start, length in [("train", 0, 50), ("val", 50, 20), ("test", 70, 20)]:
            dates = [f"2020-01-{(idx % 28) + 1:02d}" for idx in range(start, start + length)]
            values = np.arange(start, start + length, dtype="float32")
            rows = "date,value\n" + "\n".join(f"{date},{value}" for date, value in zip(dates, values)) + "\n"
            (client_dir / f"{split}.csv").write_text(rows, encoding="utf-8")

    def first_batch(seed: int):
        config = {
            "runtime": {"seed": seed},
            "data": {
                "split_dir": str(tmp_path),
                "clients": ["Nd2O3", "CeO2", "La2O3"],
                "seq_len": 4,
                "pred_len": 2,
                "batch_size": 8,
                "shuffle_train": False,
            },
        }
        train_loaders, _, _ = build_federated_loaders(config)
        x, y = next(iter(train_loaders["Nd2O3"]))
        return x.numpy().copy(), y.numpy().copy()

    first_x_a, first_y_a = first_batch(2026)
    first_x_b, first_y_b = first_batch(2027)

    np.testing.assert_allclose(first_x_a, first_x_b)
    np.testing.assert_allclose(first_y_a, first_y_b)


def test_build_server_rare_earth_evaluation_loaders_from_registration_metadata(tmp_path):
    from fedlab.datasets.rare_earth import build_server_rare_earth_evaluation_loaders

    for client, start_offset in [("Nd2O3", 0), ("CeO2", 100), ("La2O3", 200)]:
        client_dir = tmp_path / "clients" / client
        client_dir.mkdir(parents=True)
        for split, start, length in [("val", start_offset + 40, 20), ("test", start_offset + 60, 20)]:
            dates = [f"2020-01-{(idx % 28) + 1:02d}" for idx in range(start, start + length)]
            values = np.arange(start, start + length, dtype="float32")
            rows = "date,value\n" + "\n".join(f"{date},{value}" for date, value in zip(dates, values)) + "\n"
            (client_dir / f"{split}.csv").write_text(rows, encoding="utf-8")

    registration_metadata = {
        "Nd2O3": {"scale_mean": [19.5], "scale_std": [11.54339599609375]},
        "CeO2": {"scale_mean": [119.5], "scale_std": [11.54339599609375]},
        "La2O3": {"scale_mean": [219.5], "scale_std": [11.54339599609375]},
    }
    config = {
        "runtime": {"seed": 2026},
        "data": {
            "split_dir": str(tmp_path),
            "clients": ["Nd2O3", "CeO2", "La2O3"],
            "seq_len": 4,
            "pred_len": 2,
            "batch_size": 8,
        },
    }

    val_loader, test_loader = build_server_rare_earth_evaluation_loaders(config, registration_metadata=registration_metadata)

    assert val_loader is not None
    assert test_loader is not None
    assert len(val_loader) > 0
    assert len(test_loader) > 0
    assert getattr(val_loader.loaders[0], "scaler", None) is not None
    first_x, _ = next(iter(val_loader.loaders[0]))
    first_value = float(first_x[0, 0, 0])
    expected_raw = float(
        build_server_rare_earth_evaluation_loaders.__globals__['read_value_frame'](tmp_path / 'clients' / 'Nd2O3' / 'val.csv')['value'].iloc[0]
    )
    expected_value = np.float32((expected_raw - 19.5) / 11.54339599609375)
    assert first_value == pytest.approx(float(expected_value))


def test_build_federated_loaders_from_split_dir_without_test_splits(tmp_path):
    from fedlab.datasets.rare_earth import build_federated_loaders

    for client in ["Nd2O3", "CeO2", "La2O3"]:
        client_dir = tmp_path / "clients" / client
        client_dir.mkdir(parents=True)
        for split, start, length in [("train", 0, 40), ("val", 40, 30)]:
            dates = [f"2020-01-{(idx % 28) + 1:02d}" for idx in range(start, start + length)]
            values = np.arange(start, start + length, dtype="float32")
            rows = "date,value\n" + "\n".join(f"{date},{value}" for date, value in zip(dates, values)) + "\n"
            (client_dir / f"{split}.csv").write_text(rows, encoding="utf-8")
    config = {"data": {"split_dir": str(tmp_path), "clients": ["Nd2O3", "CeO2", "La2O3"], "seq_len": 4, "pred_len": 2, "batch_size": 8}}

    train_loaders, val_loader, test_loader = build_federated_loaders(config)

    assert set(train_loaders) == {"Nd2O3", "CeO2", "La2O3"}
    assert len(val_loader) > 0
    assert test_loader is None
