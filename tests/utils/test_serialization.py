import torch

from federated_ts.modeling.forecasting import build_model
from federated_ts.utils.serialization import (
    clip_state_update,
    compress_randomk,
    compress_topk,
    decompress_topk,
    dequantize_state_update,
    load_serialized,
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


def test_absmax_int8_quantization_tracks_scale_and_restores_approximately():
    update = {"w": torch.tensor([1.0, -2.0, 3.5, -0.25], dtype=torch.float32)}
    quantized = quantize_state_update(update, dtype="int8")
    restored = dequantize_state_update(quantized)

    assert quantized["w"].dtype == torch.int8
    assert "w.__scale__" in quantized
    assert state_num_bytes(quantized) < state_num_bytes(update)
    assert restored["w"].dtype == torch.float32
    assert torch.allclose(restored["w"], update["w"], atol=0.05)


def test_absmax_int8_stochastic_quantization_is_seeded_and_approximately_restores():
    update = {"w": torch.tensor([0.1, 0.9, -1.7, 3.2], dtype=torch.float32)}
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    quantized_a = quantize_state_update(update, dtype="int8", stochastic_rounding=True, generator=generator_a)
    quantized_b = quantize_state_update(update, dtype="int8", stochastic_rounding=True, generator=generator_b)
    restored = dequantize_state_update(quantized_a)

    assert torch.equal(quantized_a["w"], quantized_b["w"])
    assert torch.allclose(restored["w"], update["w"], atol=0.1)


def test_load_serialized_accepts_legacy_patchtst_wrapper_prefixes():
    config = {
        "data": {"seq_len": 21, "pred_len": 7},
        "model": {"name": "patchtst", "channels": 1, "patch_len": 7, "stride": 4, "d_model": 16, "n_heads": 4, "e_layers": 1, "d_ff": 32},
    }
    model = build_model(config)
    state = serialize_model(model)
    legacy_state = {f"model.{name}": tensor.clone() for name, tensor in state.items()}

    reloaded = build_model(config)
    load_serialized(reloaded, legacy_state)

    for name, tensor in reloaded.state_dict().items():
        assert torch.allclose(tensor, state[name])
