# Communication Compression and Privacy Notes

## Literature Scope

FedAWARE is kept in this repository only as a supplementary Xu-related adaptive aggregation baseline. For the rare-earth compression-plus-security experiments, the default comparison path is the SoteriaFL-style private random-k upload algorithm below, because it directly changes both communication volume and leakage resistance.


## Supplementary Xu et al. Baseline

The `fedaware` algorithm path implements a FedAWARE-style adaptive weighted aggregation step:

1. Each client receives the dense global model and performs standard local training.
2. Each client uploads its dense local model state.
3. The server converts each local model into a dense update `local_state - global_state`.
4. The server solves a simplex-constrained adaptive weighting problem over the client updates.
5. The adaptive weights are blended with the standard sample-count FedAvg prior using `fedaware.alpha`.
6. The weighted dense update is applied to the global model.

This keeps the training contract close to FedAvg and is preserved as a supplementary Xu et al. adaptive aggregation baseline, but it is no longer the repository's default compression/privacy comparison run.

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
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/soteriafl.yaml
```

Run a short smoke test with overrides:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/soteriafl.yaml --override federated.rounds=2 --override attack.frequency_rounds=1 --override attack.steps=1
```

## Extra Top-k Compression Baseline

The local `sparse_fedavg` / `compressed_fedavg` path remains available as an additional communication-compression baseline. It implements the core Top-k update upload idea: each client sends the largest-magnitude `topk_fraction` coordinates plus their indices, and the server reconstructs sparse updates before FedAvg-weighted aggregation. This path is kept for compression/attack analysis, but it is not the repository's Xu et al. comparison algorithm.

Run the rawdata2 full PatchTST Top-k experiment from `src`:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/topk.yaml
```

The default attack configuration now evaluates DLG and iDLG every federated round with 300 optimization steps on `cuda:0`.
