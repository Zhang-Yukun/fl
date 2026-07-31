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
