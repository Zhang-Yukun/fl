# Communication Compression and Privacy Notes

## Literature Scope

FedLab, with Zenglin Xu listed among the authors, is a flexible federated learning framework that exposes modular server, client, communication, and compression components. It is useful as a framework reference for communication-compression extensibility. The concrete privacy/compression algorithm implemented in this repository is the SoteriaFL-style component below, not claimed as a Zenglin Xu paper.

## Implemented Algorithm Component

The `soteriafl` algorithm path implements a model-update adaptation of the SoteriaFL CDP-SGD/random-k upload component:

1. Each client receives the dense global model parameters.
2. Each client trains locally and computes `local_state - global_state`.
3. The full update is clipped by global L2 norm `privacy.clip_norm`.
4. Gaussian local-DP noise with standard deviation `privacy.noise_multiplier * privacy.clip_norm` is added coordinate-wise.
5. The noisy update is compressed with unbiased random-k sparsification. Selected coordinates are scaled by `d / k`, where `d` is the total coordinate count and `k = floor(d * federated.topk_fraction)`.
6. The server decompresses sparse updates and aggregates them with sample-count FedAvg weights.

This preserves the framework's FedAvg aggregation contract while allowing direct measurement of upload compression ratio and DLG/iDLG attack outcomes. It does not implement SoteriaFL's full shifted/memory recursion.

## Recorded Metrics

Each round records client names, payload type, compressor, upload/download bytes, upload/download parameter counts, dense FedAvg reference upload size, upload compression ratio, total communication ratio, runtime, validation metrics, and attack results. `attack_results.json` stores DLG/iDLG reconstruction metrics and success flags.

## Reproduction

Run a quick SoteriaFL experiment from `src`:

```bash
conda run -n torch_env python -m scripts.train --config configs/rawdata2_soteriafl.yaml
```

Run a short smoke test with overrides:

```bash
conda run -n torch_env python -m scripts.train --config configs/rawdata2_soteriafl.yaml --override federated.rounds=2 --override attack.frequency_rounds=1 --override attack.steps=1
```
