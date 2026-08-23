from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from fedlab.modeling.ega import (
    EgaAutoEncoder,
    decode_mean_encoded_payload,
    dequantize_block_vector,
    dequantize_encoded_blocks,
    encode_state_update,
    pack_flat_blocks,
    pretrain_ega_codec,
    resolve_ega_artifact_path,
    stochastic_quantize_block_vector,
)


def test_stochastic_quantize_block_vector_stays_in_integer_domain():
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    vector = torch.tensor([-2.0, -0.25, 0.0, 0.5, 2.0], dtype=torch.float32)

    quantized = stochastic_quantize_block_vector(
        vector,
        quantization_level=4,
        normalization=2.0,
        generator=generator,
    )

    assert quantized.dtype == torch.float32
    assert torch.all(quantized <= 4)
    assert torch.all(quantized >= -4)
    assert torch.allclose(quantized, torch.round(quantized))
    restored = dequantize_block_vector(quantized, quantization_level=4, normalization=2.0)
    assert restored.shape == vector.shape


def test_encode_decode_mean_payload_restores_state_shape():
    codec = EgaAutoEncoder(block_size=4, encoded_dim=4, hidden_dim=8, residual_blocks=0)
    update = OrderedDict(
        {
            "weight": torch.tensor([[0.2, -0.2], [0.1, -0.1]], dtype=torch.float32),
            "bias": torch.tensor([0.05, -0.05], dtype=torch.float32),
        }
    )
    payload = encode_state_update(
        update,
        codec,
        quantization_level=8,
        normalization=1.0,
        block_size=4,
        contribution_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(3),
        encoded_dtype="int8",
    )

    decoded = decode_mean_encoded_payload([payload], codec)

    assert payload.encoded_blocks.dtype == torch.int8
    assert payload.encoded_scale is not None
    assert dequantize_encoded_blocks(payload).shape[1] == codec.encoded_dim
    assert list(decoded.keys()) == ["weight", "bias"]
    assert decoded["weight"].shape == update["weight"].shape
    assert decoded["bias"].shape == update["bias"].shape


def test_pack_flat_blocks_pads_to_block_size():
    flat = torch.arange(10, dtype=torch.float32)

    blocks, pad = pack_flat_blocks(flat, 4)

    assert blocks.shape == (3, 4)
    assert pad == 2


def test_pretrain_ega_codec_supports_early_stopping(tmp_path):
    config = {
        "runtime": {"seed": 2026, "device": "cpu"},
        "ega": {
            "block_size": 4,
            "encoded_dim": 4,
            "hidden_dim": 8,
            "residual_blocks": 0,
            "quantization_level": 8,
            "pretrain": {
                "device": "cpu",
                "epochs": 5,
                "patience": 1,
                "min_delta": 1e9,
                "batch_size": 4,
                "lr": 1e-3,
                "train_groups": 8,
                "val_groups": 4,
                "seed": 2026,
            },
        },
    }

    output_path = tmp_path / "ega_codec.pt"
    pretrain_ega_codec(config, num_clients=3, device=torch.device("cpu"), output_path=output_path)
    checkpoint = torch.load(output_path, map_location="cpu")

    assert checkpoint["best_epoch"] >= 0
    assert checkpoint["completed_epochs"] < config["ega"]["pretrain"]["epochs"]
    assert checkpoint["stopped_early"] is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this regression test")
def test_encode_state_update_moves_blocks_to_codec_device():
    codec = EgaAutoEncoder(block_size=4, encoded_dim=4, hidden_dim=8, residual_blocks=0).to(torch.device("cuda:0"))
    update = OrderedDict({"weight": torch.tensor([[0.2, -0.2], [0.1, -0.1]], dtype=torch.float32)})

    payload = encode_state_update(
        update,
        codec,
        quantization_level=8,
        normalization=1.0,
        block_size=4,
        contribution_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(3),
        encoded_dtype="int8",
    )

    assert payload.encoded_blocks.device.type == "cpu"
    assert payload.encoded_blocks.shape[1] == codec.encoded_dim
    assert payload.encoded_blocks.dtype == torch.int8


def test_resolve_ega_artifact_path_defaults_to_experiment_output_dir():
    config = {
        "experiment": {"output_dir": "outputs/demo_run", "name": "demo"},
        "ega": {},
    }

    path = resolve_ega_artifact_path(config, num_clients=3)

    assert path == Path("outputs/demo_run/artifacts/ega/pretrained_codec.pt")
