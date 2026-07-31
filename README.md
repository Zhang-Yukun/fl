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

- `federated_ts.utils.config`: nested YAML loading and CLI overrides.
- `federated_ts.utils.artifacts`: experiment parameter artifacts in YAML/JSON/TOML.
- `federated_ts.utils.serialization`: ordered parameter serialization and sparse updates.
- `federated_ts.datasets.rare_earth`: rare-earth CSV loading and sliding-window datasets.
- `federated_ts.modeling.forecasting`: forecasting model registry and implementations.
- `federated_ts.engine.training`: local train/evaluate loops.
- `federated_ts.federated.algorithms`: centralized training, FedAvg, compressed FedAvg.
- `federated_ts.federated.client` and `federated_ts.federated.server`: FL endpoint logic.
- `federated_ts.security.attacks`: DLG and iDLG-style gradient reconstruction attacks.
- `federated_ts.communication.grpc_service`: gRPC transport helpers for multi-process training.


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

## Test Layout

Tests mirror the package structure so each functional area has a matching test
folder:

- `tests/utils/` for config, artifacts, metrics, and serialization.
- `tests/datasets/` for rare-earth data loading and window datasets.
- `tests/modeling/` for forecasting model construction.
- `tests/federated/` for FedAvg and compressed FedAvg behavior.
- `tests/docs/` for docstring coverage checks.

## Rawdata2 PatchTST Experiment

Prepare the latest CBC rawdata2 Excel files into chronological 8:1:1 splits:

```bash
python -m scripts.prepare_rawdata2 \
  --raw-dir ../Time-Series-Prediction/dataset/data_preprocess/rawdata2 \
  --output-dir ../data/rare_earth_rawdata2
```

Run each rawdata2 experiment separately with 500 maximum epochs/rounds
and early stopping patience 50:

```bash
bash scripts/run_rawdata2_centralized.sh
bash scripts/run_rawdata2_fedavg.sh
bash scripts/run_rawdata2_fedlab_topk.sh
```

Extra overrides can be appended to any script, for example
`--override federated.rounds=30`. Generated data is stored under
`../data/rare_earth_rawdata2`; experiment artifacts are stored under the
corresponding `outputs/rawdata2_*` directory.
