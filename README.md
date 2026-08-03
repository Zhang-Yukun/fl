# Federated Rare-Earth Time-Series Framework

This package implements a three-client federated learning framework for Nd2O3,
CeO2, and La2O3 price forecasting.

## Quick Start

```bash
cd src
conda run -n torch_env python -m scripts.train --config configs/test.yaml
conda run -n torch_env python -m pytest tests
```

The default smoke config uses compressed FedAvg with top-k sparse updates. The
main rawdata2 comparison scripts are now separated into centralized training,
standard FedAvg, and the compression-plus-security SoteriaFL baseline.

## Main Modules

- `federated_ts.utils.config`: nested YAML loading and CLI overrides.
- `federated_ts.utils.artifacts`: experiment parameter artifacts in YAML/JSON/TOML.
- `federated_ts.utils.serialization`: ordered parameter serialization and sparse updates.
- `federated_ts.datasets.rare_earth`: rare-earth CSV loading and sliding-window datasets.
- `federated_ts.modeling.forecasting`: forecasting model registry and implementations.
- `federated_ts.engine.training`: local train/evaluate loops.
- `federated_ts.federated.algorithms`: centralized training, FedAvg, FedAWARE, and compressed FedAvg.
- `federated_ts.federated.client` and `federated_ts.federated.server`: FL endpoint logic.
- `federated_ts.utils.aggregation`: adaptive aggregation helpers for Xu et al.'s FedAWARE weighting.
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

Prepare the preferred `XT_data` CSV files into chronological client splits.
`new_train.csv` is split in time order into train/validation data, and
`test1.csv` through `test9.csv` are preserved as test windows and merged into
the framework's default `test.csv`:

```bash
python -m scripts.prepare_xt_data \
  --input-dir ../Time-Series-Prediction/dataset/XT_data \
  --output-dir ../data/rare_earth_rawdata2
```

The older CBC Excel rawdata2 preparation entry point remains available as
`python -m scripts.prepare_rawdata2` when `XT_data` is not available.

Run each rawdata2 experiment separately with 500 maximum epochs/rounds
and early stopping patience 50:

```bash
bash scripts/run_rawdata2_centralized.sh
bash scripts/run_rawdata2_fedavg.sh
bash scripts/run_rawdata2_soteriafl.sh
bash scripts/run_rawdata2_fedpetuning.sh
```

Extra overrides can be appended to any script, for example
`--override federated.rounds=30`. Generated data is stored under
`../data/rare_earth_rawdata2`; experiment artifacts are stored under the
corresponding `outputs/rawdata2_*` directory. `configs/rawdata2_soteriafl.yaml` is the default compression/privacy comparison config; `configs/rawdata2_fedaware.yaml` remains available as a supplementary Xu-related adaptive aggregation baseline.

An additional Xu-related communication-efficiency supplement is available in
`configs/rawdata2_fedpetuning.yaml`, which uses a FedPETuning-style frozen
PatchTST backbone with small trainable adapter/head parameters.
