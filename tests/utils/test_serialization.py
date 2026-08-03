import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import (
    clip_state_update,
    compress_randomk,
    compress_topk,
    decompress_topk,
    dequantize_state_update,
    quantize_state_update,
    privatize_sparse_update,
    privatize_state_update,
    serialize_model,
    state_num_bytes,
    subtract_state,
)


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



def test_privatize_sparse_update_keeps_shape_and_support():
    config = {"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}}
    update = subtract_state(serialize_model(build_model(config)), serialize_model(build_model(config)))
    sparse = compress_topk(update, 0.1)
    private = privatize_sparse_update(sparse, clip_norm=1.0, noise_multiplier=0.0)
    assert private.indices.equal(sparse.indices)
    assert private.total_numel == sparse.total_numel
    assert private.nbytes == sparse.nbytes


def test_randomk_compression_keeps_payload_sparse_and_scaled():
    config = {"data": {"seq_len": 4, "pred_len": 2}, "model": {"name": "mlp", "channels": 1, "hidden_size": 8}}
    update = subtract_state(serialize_model(build_model(config)), serialize_model(build_model(config)))
    generator = torch.Generator().manual_seed(7)
    sparse = compress_randomk(update, 0.05, generator=generator)
    recovered = decompress_topk(sparse)
    assert sparse.values.numel() == max(1, int(sparse.total_numel * 0.05))
    assert sparse.indices.unique().numel() == sparse.indices.numel()
    assert recovered.keys() == update.keys()
    assert state_num_bytes(update) / sparse.nbytes >= 6.0


def test_privatize_state_update_clips_before_noise():
    update = {"w": torch.tensor([3.0, 4.0])}
    clipped = clip_state_update(update, clip_norm=1.0)
    assert torch.isclose(torch.linalg.vector_norm(clipped["w"]), torch.tensor(1.0))
    private = privatize_state_update(update, clip_norm=1.0, noise_multiplier=0.0)
    assert torch.allclose(private["w"], clipped["w"])


def test_quantize_state_update_reduces_payload_size_and_restores_float32():
    update = {"w": torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32)}
    quantized = quantize_state_update(update, dtype="float16")
    restored = dequantize_state_update(quantized)

    assert quantized["w"].dtype == torch.float16
    assert restored["w"].dtype == torch.float32
    assert state_num_bytes(quantized) < state_num_bytes(update)
    assert torch.allclose(restored["w"], update["w"], atol=1e-3)
