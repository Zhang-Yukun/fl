# Federated Rare-Earth Time-Series Framework

This package implements a three-client federated learning framework for Nd2O3,
CeO2, and La2O3 price forecasting.

## Quick Start

```bash
cd src
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/test.yaml
conda run -n torch_env python -m pytest tests
```

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
- `fedlab.utils.aggregation`: adaptive aggregation helpers.
- `fedlab.security.attacks`: DLG and iDLG-style reconstruction attacks over server-visible update payloads, with replay support for saved updates.
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

## Script Layout

The `scripts/` directory keeps only three categories of launchers:

- suite scripts such as `run_oracle_suite.sh`, `run_formal_*.sh`, and `run_suite_500r_*.sh`
- oracle wrapper scripts such as `run_oracle_gpu0_batch.sh`
- EGA sweep scripts such as `run_ega_*` and `run_rawdata2_ega_sweep_*.sh`

Per-algorithm one-off launchers and older repro/test-matrix shell wrappers were removed.
Use the suite scripts or call the Python entrypoints directly when you need a custom run.
