import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import compress_topk, decompress_topk, serialize_model, state_num_bytes, subtract_state


def test_state_serialization_and_sparse_ratio():
    config = {"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}}
    model_a = build_model(config)
    model_b = build_model(config)
    state_a = serialize_model(model_a)
    state_b = serialize_model(model_b)
    update = subtract_state(state_b, state_a)
    sparse = compress_topk(update, 0.05)
    recovered = decompress_topk(sparse)
    assert recovered.keys() == update.keys()
    assert state_num_bytes(update) / sparse.nbytes >= 6.0
    assert all(torch.isfinite(tensor).all() for tensor in recovered.values())

