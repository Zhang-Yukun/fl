from collections import OrderedDict

import torch

from fedlab.modeling.ega import (
    EgaAutoEncoder,
    decode_mean_encoded_payload,
    dequantize_block_vector,
    encode_state_update,
    pack_flat_blocks,
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
    )

    decoded = decode_mean_encoded_payload([payload], codec)

    assert list(decoded.keys()) == ["weight", "bias"]
    assert decoded["weight"].shape == update["weight"].shape
    assert decoded["bias"].shape == update["bias"].shape


def test_pack_flat_blocks_pads_to_block_size():
    flat = torch.arange(10, dtype=torch.float32)

    blocks, pad = pack_flat_blocks(flat, 4)

    assert blocks.shape == (3, 4)
    assert pad == 2
