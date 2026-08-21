# Output Artifact Field Reference

This document describes the output files produced by the current codebase and the exact scalar/image keys logged to wandb.

Covered modes:
- Centralized training
- Single-process federated training
- gRPC multi-process / multi-node federated training

Source of truth:
- `fedlab/federated/algorithms.py`
- `fedlab/federated/server.py`
- `fedlab/communication/grpc_training.py`
- `fedlab/security/attacks.py`
- `fedlab/utils/tracking.py`

## 1. Common output files

| File | Meaning | Modes |
| --- | --- | --- |
| `summary.json` | Final compact summary for experiment comparison | all |
| `metrics.json` | Training history; object for centralized, round list for federated/gRPC | all |
| `attack_results.json` | Per-attack detailed records | federated/gRPC with attacks enabled |
| `model.pt` | Final global model parameters | federated/gRPC |
| `centralized_model.pt` | Final centralized model parameters | centralized |
| `config.yaml` / `config.json` / `config.toml` | Saved effective config snapshot | all |
| `run.log` | Runtime log file | all |
| `attack_artifacts/` | Saved reconstruction tensors for attacks | federated/gRPC with attacks enabled |
| `snapshots/round_xxxx/` | Periodic intermediate snapshots | when enabled |

## 2. Communication semantics

### 2.1 `parameter_*`

`parameter_*` now means **algorithm-effective communication bytes**:
- model parameters or updates
- any auxiliary information required to reconstruct or aggregate them
- examples: sparse indices, quantization scale, EGA encoded blocks, EGA download context required for reconstruction

It does **not** include pure transport-layer overhead such as gRPC/protobuf/pickle/envelope/control flags.

### 2.2 `transport_*`

`transport_*` means **actual serialized transport bytes**:
- includes `parameter_*`
- plus envelope, RPC serialization, control fields, etc.

Use:
- `parameter_*` for algorithm communication analysis
- `transport_*` for deployment/network-cost analysis

### 2.3 Alias fields

In the current implementation:
- `download_bytes == parameter_download_bytes`
- `upload_bytes == parameter_upload_bytes`
- `total_download_bytes == total_parameter_download_bytes`
- `total_upload_bytes == total_parameter_upload_bytes`

These alias fields also follow the new **algorithm-effective communication** semantics.

## 3. `summary.json`

### 3.1 Centralized `summary.json`

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | final test metrics |
| `test.mse` | float | test MSE |
| `test.mae` | float | test MAE |
| `test.mape` | float | test MAPE |
| `rounds` | int | recorded training rounds |
| `total_time_seconds` | float | total wall-clock time |
| `best_round` | int | best validation round |
| `best_val_mse` | float | best validation MSE |
| `best_val_mae` | float | best validation MAE |
| `best_val_mape` | float | best validation MAPE |
| `test_checkpoint` | string | current value is `best_validation` |

### 3.2 Federated / gRPC `summary.json`

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | final test metrics under the active evaluation scope |
| `test.mse` | float | test MSE |
| `test.mae` | float | test MAE |
| `test.mape` | float | test MAPE |
| `active_test_scope` | string | scope used by `test`, e.g. `protocol` or `oracle_full_update` |
| `evaluation_mode` | string | configured evaluation mode |
| `best_val_scope` | string | validation scope used for best checkpoint selection |
| `protocol_test` | object | protocol-scope test metrics |
| `oracle_test` | object | oracle-scope test metrics; equals protocol view when oracle mode is not enabled |
| `rounds` | int | recorded federated rounds |
| `total_time_seconds` | float | total wall-clock time |
| `best_round` | int | best validation round |
| `best_val_mse` | float | best validation MSE |
| `best_val_mae` | float | best validation MAE |
| `best_val_mape` | float | best validation MAPE |
| `test_checkpoint` | string | current value is `best_validation` |
| `last_parameter_upload_compression_ratio` | float | last-round `fedavg_reference_upload_bytes / last_parameter_upload_bytes` |
| `last_parameter_total_communication_ratio` | float | last-round `fedavg_reference_total_bytes / last_parameter_total_bytes` |
| `last_upload_compression_ratio` | float | compatibility alias of `last_parameter_upload_compression_ratio` |
| `last_total_communication_ratio` | float | compatibility alias of `last_parameter_total_communication_ratio` |
| `last_communication_ratio` | float | compatibility alias of `last_upload_compression_ratio` |
| `last_parameter_upload_bytes` | int | cumulative algorithm-effective upload bytes in the last round |
| `last_parameter_download_bytes` | int | cumulative algorithm-effective download bytes in the last round |
| `last_parameter_total_bytes` | int | last-round parameter total bytes |
| `last_transport_upload_bytes` | int | last-round actual serialized upload bytes |
| `last_transport_download_bytes` | int | last-round actual serialized download bytes |
| `last_transport_total_bytes` | int | last-round transport total bytes |
| `last_transport_upload_overhead_bytes` | int | last-round upload transport overhead |
| `last_transport_download_overhead_bytes` | int | last-round download transport overhead |
| `last_transport_upload_compression_ratio` | float | last-round `fedavg_reference_upload_bytes / last_transport_upload_bytes` |
| `last_transport_total_communication_ratio` | float | last-round `fedavg_reference_total_bytes / last_transport_total_bytes` |
| `total_parameter_upload_bytes` | int | cumulative algorithm-effective upload bytes |
| `total_parameter_download_bytes` | int | cumulative algorithm-effective download bytes |
| `total_parameter_bytes` | int | cumulative algorithm-effective total bytes |
| `total_transport_upload_bytes` | int | cumulative actual upload transport bytes |
| `total_transport_download_bytes` | int | cumulative actual download transport bytes |
| `total_transport_bytes` | int | cumulative actual transport total bytes |
| `total_transport_upload_overhead_bytes` | int | cumulative upload transport overhead |
| `total_transport_download_overhead_bytes` | int | cumulative download transport overhead |
| `attack_target_type` | string | attack interception target type |
| `attack_primary_metric` | string | primary attack metric name |
| `attack_primary_metric_direction` | string | current value is `higher_is_more_private` |
| `attack_overall_avg_primary_metric` | float or null | average primary attack metric |
| `attack_overall_best_primary_metric` | float or null | best (minimum) primary attack metric |
| `attack_overall_avg_mse` | float or null | compatibility alias of `attack_overall_avg_primary_metric` |
| `attack_success_rate` | float | overall attack success rate |
| `attack_evaluations` | int | number of attack records |
| `attack_summary` | object | aggregated attack summary |
| `privacy_accountant` | string or null | privacy accountant name |
| `privacy_epsilon` | float or null | epsilon |
| `privacy_delta` | float or null | delta |
| `privacy_rdp_alpha` | float or null | RDP alpha |
| `privacy_rdp_total` | float or null | cumulative RDP |
| `privacy_sampling_rate` | float or null | sampling rate |
| `adaptive_clip_norm` | float or null | adaptive clip norm |
| `adaptive_clip_median_norm` | float or null | median update norm |
| `adaptive_reference_clip_norm` | float or null | reference clip norm |
| `adaptive_noise_std` | float or null | noise std |
| `privacy_trust_model` | string or null | current possible value: `central_dp_trusted_aggregator` |
| `transport` | string | only in gRPC summaries; current value `grpc` |

## 3.3 `summary.json.attack_summary`

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_metric` | string | primary metric name |
| `primary_metric_direction` | string | current value `higher_is_more_private` |
| `target_type` | string or null | attack target type |
| `success_rate_threshold` | float | threshold used for `passes` |
| `overall_avg_primary_metric` | float or null | average primary metric |
| `overall_best_primary_metric` | float or null | best (minimum) primary metric |
| `overall_avg_mse` | float or null | compatibility alias |
| `overall_best_mse` | float or null | compatibility alias |
| `overall_avg_exact_target_mse` | float or null | average exact-target reconstruction error |
| `overall_avg_nearest_client_train_mse` | float or null | average nearest-train-sample error |
| `overall_avg_psnr` | float or null | average PSNR |
| `overall_avg_ssim` | float or null | average SSIM |
| `overall_avg_objective_mse` | float or null | average objective-space error |
| `overall_avg_gradient_mse` | float or null | compatibility alias |
| `overall_success_rate` | float | overall success rate |
| `overall_success_rate_percent` | float | overall success rate in percent |
| `overall_passes` | bool | whether the overall success rate is below threshold |
| `methods` | object | grouped method summary, e.g. `DLG`, `iDLG` |

### `summary.json.attack_summary.methods.<method>`

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_metric` | string | primary metric for that method |
| `target_type` | string or null | target type |
| `success_count` | int | successful attack count |
| `total_count` | int | total attack count |
| `success_rate` | float | success rate |
| `success_rate_percent` | float | success rate in percent |
| `avg_primary_metric` | float or null | average primary metric |
| `best_primary_metric` | float or null | best (minimum) primary metric |
| `avg_mse` | float or null | compatibility alias |
| `best_mse` | float or null | compatibility alias |
| `avg_exact_target_mse` | float or null | average exact-target reconstruction error |
| `avg_nearest_client_train_mse` | float or null | average nearest-train-sample error |
| `avg_psnr` | float or null | average PSNR |
| `avg_ssim` | float or null | average SSIM |
| `best_ssim` | float or null | best SSIM |
| `avg_objective_mse` | float or null | average objective error |
| `avg_gradient_mse` | float or null | compatibility alias |
| `avg_time_seconds` | float or null | average attack time |
| `passes` | bool | whether method success rate is below threshold |

## 4. `metrics.json`

### 4.1 Centralized `metrics.json`

| Field | Type | Meaning |
| --- | --- | --- |
| `history` | list[object] | per-round history |
| `test` | object | final test metrics |
| `rounds` | int | number of recorded rounds |
| `total_time_seconds` | float | total wall-clock time |
| `best_round` | int | best validation round |
| `best_val` | object | best validation metrics |
| `test_checkpoint` | string | current value `best_validation` |

### `metrics.json.history[*]`

| Field | Type | Meaning |
| --- | --- | --- |
| `round` | int | round index |
| `train_loss` | float | training loss |
| `val_mse` | float | validation MSE |
| `val_mae` | float | validation MAE |
| `val_mape` | float | validation MAPE |
| `round_time_seconds` | float | per-round time |
| `elapsed_time_seconds` | float | cumulative elapsed time |

### 4.2 Federated / gRPC `metrics.json`

Federated/gRPC `metrics.json` is a list of `RoundRecord` objects. The round-level fields and client-level fields match the Chinese reference above exactly.

## 5. `attack_results.json`

Each item is produced by `AttackResult.to_record()` and includes:
- attack identity: `name`, `target_type`, `client_id`, `round_index`, `sample_index`
- core metrics: `mse`, `psnr`, `ssim`, `gradient_mse`, `objective_mse`
- success metadata: `success`, `success_threshold`, `metric_name`, `primary_metric_name`, `primary_metric_value`
- reconstruction metrics: `exact_target_mse`, `nearest_client_train_mse`, `nearest_client_train_indices`
- artifact link: `artifact_path`

## 6. wandb keys

### 6.1 General

When `tracker.log(..., step=...)` is used, the payload also carries:
- `tracking/step`

### 6.2 Centralized scalar keys

Startup:
- `run/model_parameters`
- `run/model_bytes`
- `run/mode`

Per round:
- `round/loss`
- `round/val_mse`
- `round/val_mae`
- `round/val_mape`
- `round/time_seconds`
- `run/elapsed_time_seconds`
- `tracking/step`

Final:
- `test/mse`
- `test/mae`
- `test/mape`
- `run/total_time_seconds`
- `run/best_round`
- `run/best_val_mse`
- `run/best_val_mae`
- `run/best_val_mape`

### 6.3 Federated / gRPC startup keys

- `run/algorithm`
- `run/client_count`
- `run/model_parameters`
- `run/model_bytes`
- `run/compressed_uploads`
- `run/transport` (gRPC startup only)

### 6.4 Federated / gRPC per-round keys

- every `RoundRecord` top-level field is flattened as `round/<field>`
- every `ClientCommunicationRecord` field is flattened as `client/<client_id>/<field>`
- cumulative communication summary fields are flattened as `cumulative/<field>`
- attack round payload fields are logged under `attack/...`

### 6.5 Final scalar keys

Single-process federated final keys:
- `test/*`
- `protocol_test/*`
- `oracle_test/*`
- `run/total_time_seconds`
- `run/evaluation_mode`
- `run/best_round`
- `run/best_val_mse`
- `run/best_val_mae`
- `run/best_val_mape`
- `privacy/epsilon`
- `privacy/delta`
- `privacy/rdp_total`
- `privacy/sampling_rate`
- `privacy/adaptive_clip_norm`

gRPC additionally logs:
- `run/rounds`
- `run/transport`

### 6.6 Image keys

Prediction plots:
- `prediction/centralized/val`
- `prediction/centralized/test`
- `prediction/federated/val_protocol`
- `prediction/federated/val_oracle`
- `prediction/federated/test_protocol`
- `prediction/federated/test_oracle`
- `prediction/grpc/val_protocol`
- `prediction/grpc/val_oracle`
- `prediction/grpc/test_protocol`
- `prediction/grpc/test_oracle`

Attack reconstructions:
- `attack/DLG/reconstruction`
- `attack/iDLG/reconstruction`

## 7. Notes

1. Use `summary.json` for final comparison.
2. Use `metrics.json` for convergence and per-round communication analysis.
3. Use `parameter_*` for algorithm communication analysis.
4. Use `transport_*` for real serialized network cost.
5. For EGA, download-side reconstruction context is already counted in `parameter_download_*`, not only in `transport_*`.
