# Federated Rare-Earth Time-Series Framework

This package implements a three-client federated learning framework for Nd2O3,
CeO2, and La2O3 price forecasting.

## Quick Start

```bash
cd src
conda run -n torch_env python -m scripts.train --config configs/test.yaml
conda run -n torch_env python -m pytest tests
```

The default experiment uses compressed FedAvg with top-k sparse updates. With
`topk_fraction: 0.05`, the measured single-round upload compression ratio is
expected to be at least 6x compared with dense FedAvg uploads.

## Main Modules

- `federated_ts.config`: nested YAML loading and CLI overrides.
- `federated_ts.data`: rare-earth CSV loading and sliding-window datasets.
- `federated_ts.serialization`: ordered parameter serialization and sparse updates.
- `federated_ts.algorithms`: centralized training, FedAvg, compressed FedAvg.
- `federated_ts.attacks`: DLG and iDLG-style gradient reconstruction attacks.
- `federated_ts.grpc_service`: gRPC transport helpers for multi-process training.


## Multi-Process gRPC

Start one server and three clients in separate shells or nodes:

```bash
python -m scripts.server --config configs/default.yaml
python -m scripts.client --client-id Nd2O3 --config configs/default.yaml
python -m scripts.client --client-id CeO2 --config configs/default.yaml
python -m scripts.client --client-id La2O3 --config configs/default.yaml
```

## Experiment Artifacts

Experiment parameters are saved as `config.yaml` by default. To save multiple
parameter formats, set `artifacts.config_formats`, for example:

```yaml
artifacts:
  config_formats: [yaml, json, toml]
```
