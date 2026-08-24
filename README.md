# Federated Rare-Earth Time-Series Framework

This package implements a three-client federated learning framework for Nd2O3,
CeO2, and La2O3 price forecasting.

## Quick Start

```bash
cd src
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/test.yaml
conda run -n torch_env python -m pytest tests
```

The default smoke config uses compressed FedAvg with top-k sparse updates. The
main rawdata2 comparison scripts are now separated into centralized training,
standard FedAvg, the retained sparse Top-k baseline, and the rest of the active eight federated algorithm configs.

For a fuller Chinese usage guide covering configuration, algorithms, gRPC,
artifacts, and extension points, see
[`docs/框架使用手册.md`](docs/框架使用手册.md).

## Main Modules

- `fedlab.utils.config`: nested YAML loading and CLI overrides.
- `fedlab.utils.artifacts`: experiment parameter artifacts in YAML/JSON/TOML.
- `fedlab.utils.serialization`: ordered parameter serialization, selective layer filtering, trainable/untrainable state export, quantization, and sparse updates.
- `fedlab.datasets.rare_earth`: rare-earth CSV loading and sliding-window datasets.
- `fedlab.modeling.forecasting`: forecasting model registry and implementations.
- `fedlab.engine.training`: local train/evaluate loops.
- `fedlab.federated.methods`: pluggable federated algorithm implementations grouped by dense, sparse, quantized, and encoded families.
- `fedlab.federated.client` and `fedlab.federated.server`: algorithm-agnostic FL endpoint runtime that delegates payload construction, aggregation, and attack-view extraction to the active method.
- `fedlab.federated.algorithms`: single-process federated orchestration, attack scheduling, and evaluation flow built on the method registry.
- `fedlab.utils.aggregation`: adaptive aggregation helpers for Xu et al.'s FedAWARE weighting.
- `fedlab.security.attacks`: DLG and iDLG-style gradient reconstruction attacks.
- `fedlab.communication.grpc_service`: gRPC transport helpers for multi-process training.

To add a new federated algorithm, create one method module under
`fedlab/federated/methods/`, register it, then add the config/script/tests.
The runtime no longer requires scattered string-branch edits across client,
server, and orchestration code.


## Multi-Process gRPC

Start one server and three clients in separate shells or nodes:

```bash
python -m fedlab.entrypoints.server --config configs/default.yaml
python -m fedlab.entrypoints.client --client-id Nd2O3 --config configs/default.yaml
python -m fedlab.entrypoints.client --client-id CeO2 --config configs/default.yaml
python -m fedlab.entrypoints.client --client-id La2O3 --config configs/default.yaml
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
python -m fedlab.tools.prepare_xt_data \
  --input-dir ../Time-Series-Prediction/dataset/XT_data \
  --output-dir ../data/rare_earth_rawdata2
```

The older CBC Excel rawdata2 preparation entry point remains available as
`python -m fedlab.tools.prepare_rawdata2` when `XT_data` is not available.

Run each rawdata2 experiment separately with 500 maximum epochs/rounds
and early stopping patience 50:

```bash
bash scripts/run_rawdata2_centralized.sh
bash scripts/run_rawdata2_fedavg.sh
bash scripts/run_rawdata2_fedlab_topk.sh
bash scripts/run_rawdata2_qsgd.sh
bash scripts/run_rawdata2_randomk.sh
bash scripts/run_rawdata2_sign.sh
bash scripts/run_rawdata2_adaptive_clipped_rdp_fedavg_deterministic.sh
bash scripts/run_rawdata2_secure_quantized_fedavg.sh
bash scripts/run_rawdata2_ega.sh
```

Extra overrides can be appended to any script, for example
`--override federated.rounds=30`. Generated data is stored under
`../data/rare_earth_rawdata2`; experiment artifacts are stored under the
corresponding `outputs/<algorithm>` directory. The retained federated algorithm configs are `configs/fedavg.yaml`, `configs/topk.yaml`, `configs/qsgd.yaml`, `configs/randomk.yaml`, `configs/sign.yaml`, `configs/adaptive_clipped_rdp_fedavg.yaml`, `configs/secure_quantized_fedavg.yaml`, and `configs/ega.yaml`. 

## Reproduce

```bash
bash scripts/run_repro_pat50_centralized.sh
bash scripts/run_repro_pat50_fedavg.sh
bash scripts/run_repro_pat50_qint8_bidir.sh
python -m fedlab.tools.compare_experiment_results outputs/repro_pat50/fedavg_seed2026_pat50_payloadv1 outputs/repro_pat50/secure_qint8_bidir_seed2026_pat50_payloadv1
```