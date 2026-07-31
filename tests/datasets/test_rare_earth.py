import numpy as np

from federated_ts.datasets.rare_earth import Standardizer, WindowDataset, split_array


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
    from federated_ts.datasets.rare_earth import build_federated_loaders

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
