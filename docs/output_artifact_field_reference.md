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
| `attack_primary_primary_metric_name` | string | primary attack metric name |
| `attack_primary_primary_metric_name_direction` | string | current value is `higher_is_more_private` |
| `attack_overall_avg_primary_metric_value_value_value` | float or null | average primary attack metric |
| `attack_overall_best_primary_metric_value_value_value` | float or null | best (minimum) primary attack metric |
| `attack_overall_avg_mse` | float or null | compatibility alias of `attack_overall_avg_primary_metric_value_value_value` |
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
| `primary_primary_metric_name` | string | primary metric name |
| `primary_metric_direction` | string | current value `higher_is_more_private` |
| `target_type` | string or null | attack target type |
| `success_rate_threshold` | float | threshold used for `passes` |
| `overall_avg_primary_metric_value_value` | float or null | average primary metric |
| `overall_best_primary_metric_value_value` | float or null | best (minimum) primary metric |
| `overall_avg_mse` | float or null | compatibility alias |
| `overall_best_mse` | float or null | compatibility alias |
| `overall_avg_exact_target_mse` | float or null | average exact-target reconstruction error |
| `overall_avg_nearest_client_train_mse` | float or null | average nearest-train-sample error |
| `overall_avg_psnr` | float or null | average PSNR |
| `overall_avg_ssim` | float or null | average SSIM |
| `overall_avg_objective_mse` | float or null | average objective-space error |
| `overall_avg_objective_mse` | float or null | compatibility alias |
| `overall_success_rate` | float | overall success rate |
| `overall_success_rate_percent` | float | overall success rate in percent |
| `overall_passes` | bool | whether the overall success rate is below threshold |
| `methods` | object | grouped method summary, e.g. `DLG`, `iDLG` |
| `clients` | object | per-client attack summary keyed by client id, e.g. `Nd2O3` |

### `summary.json.attack_summary.methods.<method>`

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_primary_metric_name` | string | primary metric for that method |
| `target_type` | string or null | target type |
| `success_count` | int | successful attack count |
| `total_count` | int | total attack count |
| `success_rate` | float | success rate |
| `success_rate_percent` | float | success rate in percent |
| `avg_primary_metric_value` | float or null | average primary metric |
| `best_primary_metric_value` | float or null | best (minimum) primary metric |
| `avg_mse` | float or null | compatibility alias |
| `best_mse` | float or null | compatibility alias |
| `avg_exact_target_mse` | float or null | average exact-target reconstruction error |
| `avg_nearest_client_train_mse` | float or null | average nearest-train-sample error |
| `avg_psnr` | float or null | average PSNR |
| `avg_ssim` | float or null | average SSIM |
| `best_ssim` | float or null | best SSIM |
| `avg_objective_mse` | float or null | average objective error |
| `avg_objective_mse` | float or null | compatibility alias |
| `avg_time_seconds` | float or null | average attack time |
| `passes` | bool | whether method success rate is below threshold |



### `summary.json.attack_summary.clients.<client_id>`

This object reuses the same schema as top-level `attack_summary`, but only for the specified client.

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_primary_metric_name` | string | primary metric name for this client |
| `primary_metric_direction` | string | current value `higher_is_more_private` |
| `target_type` | string or null | attack target type for this client |
| `success_rate_threshold` | float | threshold used for `passes` |
| `overall_avg_primary_metric_value_value` | float or null | average primary metric for this client |
| `overall_best_primary_metric_value_value` | float or null | best (minimum) primary metric for this client |
| `overall_avg_mse` | float or null | compatibility alias |
| `overall_best_mse` | float or null | compatibility alias |
| `overall_avg_exact_target_mse` | float or null | average exact-target reconstruction error for this client |
| `overall_avg_nearest_client_train_mse` | float or null | average nearest-train-sample error for this client |
| `overall_avg_psnr` | float or null | average PSNR for this client |
| `overall_avg_ssim` | float or null | average SSIM for this client |
| `overall_avg_objective_mse` | float or null | average objective error for this client |
| `overall_avg_objective_mse` | float or null | compatibility alias |
| `overall_success_rate` | float | success rate for this client |
| `overall_success_rate_percent` | float | success rate in percent |
| `overall_passes` | bool | whether the client success rate is below threshold |
| `methods` | object | method summary within this client |

### `summary.json.attack_summary.clients.<client_id>.methods.<method>`

This object uses the same fields as `summary.json.attack_summary.methods.<method>`, but restricted to one client and one attack method.

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
- core metrics: `mse`, `psnr`, `ssim`, `objective_mse`, `objective_mse`
- success metadata: `success`, `success_threshold`, `primary_metric_name`, `primary_primary_metric_name`, `primary_metric_value`
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

This namespace does not invent a second set of metrics. It flattens four existing sources into wandb-friendly keys:
- `round/...`: server-side round summary after aggregation
- `client/<client_id>/...`: per-client local loss, payload, and communication details for that round
- `cumulative/...`: communication totals accumulated from round 0 up to the current round
- `attack/...`: attack results emitted after the attack worker finishes

#### 6.4.1 `round/<RoundRecord field>`

`round/...` is the main per-round view for convergence and round-level analysis.

Read it in six groups:

1. Training and validation behavior
- `round/train_loss`
- `round/val_mse`, `round/val_mae`, `round/val_mape`
- `round/active_val_scope`
- `round/active_val_mse`, `round/active_val_mae`, `round/active_val_mape`
- `round/protocol_val_*`
- `round/oracle_val_*`

Interpretation:
- `round/val_*` is a compatibility view.
- `round/active_val_*` is the recommended source because it explicitly matches the validation scope currently used for selection.
- When oracle evaluation is enabled, `protocol_val_*` and `oracle_val_*` let you compare the protocol path against the ideal full-update view in the same round.

2. Time and round index
- `round/round`
- `round/round_time_seconds`
- `round/elapsed_time_seconds`

Interpretation:
- `round/round` starts at 0.
- `round/round_time_seconds` is the wall-clock time for that full federated round.
- `round/elapsed_time_seconds` is cumulative runtime since experiment start.

3. Current global model size
- `round/model_parameters`
- `round/model_bytes`

Interpretation:
- these describe the aggregated global model itself, not the bytes transmitted in that round.

4. Algorithm-effective communication for the current round
- `round/total_download_bytes`, `round/total_download_parameters`
- `round/total_upload_bytes`, `round/total_upload_parameters`
- `round/total_parameter_download_bytes`, `round/total_parameter_download_parameters`
- `round/total_parameter_upload_bytes`, `round/total_parameter_upload_parameters`
- `round/total_parameter_bytes`

Interpretation:
- these are the bytes that the algorithm must communicate for reconstruction/aggregation.
- `total_download_bytes` and `total_upload_bytes` are compatibility aliases.
- use this group as the primary communication-analysis view for algorithm comparisons.

5. Actual transport for the current round
- `round/total_transport_download_bytes`
- `round/total_transport_upload_bytes`
- `round/total_transport_bytes`
- `round/total_transport_download_overhead_bytes`
- `round/total_transport_upload_overhead_bytes`

Interpretation:
- these include transport-layer costs on top of the algorithm payload.
- use this group for deployment-oriented network-cost analysis.
- `overhead` means `transport - parameter`.

6. Compression, privacy, and mode metadata
- `round/fedavg_reference_*`
- `round/parameter_upload_compression_ratio`
- `round/parameter_total_communication_ratio`
- `round/upload_compression_ratio`, `round/total_communication_ratio`, `round/communication_ratio`
- `round/transport_upload_compression_ratio`
- `round/transport_total_communication_ratio`
- `round/privacy_*`
- `round/adaptive_*`
- `round/evaluation_mode`
- `tracking/step`

Interpretation:
- `fedavg_reference_*` is the dense FedAvg baseline used to compute compression ratios.
- `parameter_*_ratio` is the algorithm-level compression view.
- `transport_*_ratio` is the real transport compression view.
- `privacy/*` values are meaningful only for methods that actually track privacy.
- `tracking/step` mirrors the round index and keeps asynchronous logging aligned.

#### 6.4.2 `client/<client_id>/<ClientCommunicationRecord field>`

`client/<client_id>/...` is the per-client view for one round. Use it to answer questions like:
- which client uploaded the most?
- which client had the highest local loss?
- which payload semantic was used by a given client?

Read it in five groups:

1. Local training and aggregation semantics
- `num_samples`
- `loss`
- `aggregation_payload_kind`
- `evaluation_payload_kind`
- `aggregation_weight`

2. Algorithm-effective download
- `download_bytes`, `download_parameters`
- `parameter_download_bytes`, `parameter_download_parameters`
- `dense_download_reference_bytes`, `dense_download_reference_parameters`

3. Algorithm-effective upload
- `upload_bytes`, `upload_parameters`
- `parameter_upload_bytes`, `parameter_upload_parameters`
- `dense_upload_reference_bytes`, `dense_upload_reference_parameters`

4. Actual transport
- `transport_download_bytes`
- `transport_upload_bytes`
- `transport_download_overhead_bytes`
- `transport_upload_overhead_bytes`

5. Compression/privacy metadata
- `compressor`
- `privacy_clip_norm`
- `privacy_noise_multiplier`

Interpretation:
- `parameter_*` is the payload required by the algorithm.
- `transport_*` adds transport-layer overhead.
- `dense_*_reference_*` fields provide the dense FedAvg baseline for the same client.

#### 6.4.3 `cumulative/<communication summary field>`

`cumulative/...` is not “what happened in this round”; it is “how much has happened up to this round”.

Use it for:
- total communication vs round plots
- comparing algorithms at the same round budget
- checking whether an algorithm saves communication early or only later

Key fields:
- `cumulative/total_parameter_upload_bytes`
- `cumulative/total_parameter_download_bytes`
- `cumulative/total_parameter_bytes`
- `cumulative/total_transport_upload_bytes`
- `cumulative/total_transport_download_bytes`
- `cumulative/total_transport_bytes`
- plus `last_*` snapshots and corresponding ratio/overhead keys

Interpretation:
- `total_*` means accumulated from round 0 to the current round.
- `last_*` means the most recently completed round only.

#### 6.4.4 `attack/...`

`attack/...` is emitted after an attack batch completes. It is the privacy-analysis channel rather than the training-performance channel.

There are two levels:

1. Aggregate attack status for the current attack batch
- `attack/round_index`
- `attack/time_seconds`
- `attack/evaluations_this_round`
- `attack/clients_this_round`
- `attack/samples_per_client`
- `attack/primary_primary_metric_name`
- `attack/overall_avg_primary_metric_value_value_so_far`
- `attack/overall_avg_mse_so_far`
- `attack/success_rate_so_far`

2. Method-specific attack metrics for `<method>` such as `DLG` or `iDLG`
- `attack/<method>/primary_primary_metric_name`
- `attack/<method>/primary_primary_metric_name`
- `attack/<method>/mse`
- `attack/<method>/reconstruction_mse`
- `attack/<method>/avg_primary_metric_value_so_far`
- `attack/<method>/avg_mse_so_far`
- `attack/<method>/psnr`
- `attack/<method>/ssim`
- `attack/<method>/iterations`
- `attack/<method>/time_seconds`
- `attack/<method>/objective_mse`
- `attack/<method>/objective_mse`
- `attack/<method>/success`
- `attack/<method>/success_rate_so_far`

3. Client-specific merged attack metrics for `<client_id>`
- `attack/client/<client_id>/primary_primary_metric_name`
- `attack/client/<client_id>/primary_primary_metric_name`
- `attack/client/<client_id>/mse`
- `attack/client/<client_id>/reconstruction_mse`
- `attack/client/<client_id>/avg_primary_metric_value_so_far`
- `attack/client/<client_id>/avg_mse_so_far`
- `attack/client/<client_id>/psnr`
- `attack/client/<client_id>/ssim`
- `attack/client/<client_id>/iterations`
- `attack/client/<client_id>/time_seconds`
- `attack/client/<client_id>/objective_mse`
- `attack/client/<client_id>/objective_mse`
- `attack/client/<client_id>/success`
- `attack/client/<client_id>/success_rate_so_far`

4. Client-and-method-specific attack metrics for `<client_id>` and `<method>`
- `attack/client/<client_id>/<method>/primary_primary_metric_name`
- `attack/client/<client_id>/<method>/primary_primary_metric_name`
- `attack/client/<client_id>/<method>/mse`
- `attack/client/<client_id>/<method>/reconstruction_mse`
- `attack/client/<client_id>/<method>/avg_primary_metric_value_so_far`
- `attack/client/<client_id>/<method>/avg_mse_so_far`
- `attack/client/<client_id>/<method>/psnr`
- `attack/client/<client_id>/<method>/ssim`
- `attack/client/<client_id>/<method>/iterations`
- `attack/client/<client_id>/<method>/time_seconds`
- `attack/client/<client_id>/<method>/objective_mse`
- `attack/client/<client_id>/<method>/objective_mse`
- `attack/client/<client_id>/<method>/success`
- `attack/client/<client_id>/<method>/success_rate_so_far`

Interpretation:
- “how hard is it to attack right now?”: use `attack/<method>/primary_primary_metric_name` or `attack/client/<client_id>/primary_primary_metric_name`
- “is privacy improving or degrading over time?”: use `attack/<method>/avg_primary_metric_value_so_far` or `attack/client/<client_id>/avg_primary_metric_value_so_far`
- “is the attack succeeding less often?”: use `attack/<method>/success_rate_so_far` or `attack/client/<client_id>/success_rate_so_far`
- “how good is the reconstruction this round?”: inspect `primary_primary_metric_name`, `psnr`, and `ssim` together
- “which client is easier to attack?”: compare `attack/client/<client_id>/...` across clients

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
- `prediction/centralized/val/client/<client_id>`
- `prediction/centralized/test/client/<client_id>`
- `prediction/federated/val_protocol`
- `prediction/federated/val_oracle`
- `prediction/federated/test_protocol`
- `prediction/federated/test_oracle`
- `prediction/federated/val_protocol/client/<client_id>`
- `prediction/federated/val_oracle/client/<client_id>`
- `prediction/federated/test_protocol/client/<client_id>`
- `prediction/federated/test_oracle/client/<client_id>`
- `prediction/grpc/val_protocol`
- `prediction/grpc/val_oracle`
- `prediction/grpc/test_protocol`
- `prediction/grpc/test_oracle`
- `prediction/grpc/val_protocol/client/<client_id>`
- `prediction/grpc/val_oracle/client/<client_id>`
- `prediction/grpc/test_protocol/client/<client_id>`
- `prediction/grpc/test_oracle/client/<client_id>`

Attack reconstructions:
- `attack/DLG/reconstruction`
- `attack/iDLG/reconstruction`
- `attack/client/<client_id>/DLG/reconstruction`
- `attack/client/<client_id>/iDLG/reconstruction`

## 7. Notes

1. Use `summary.json` for final comparison.
2. Use `metrics.json` for convergence and per-round communication analysis.
3. Use `parameter_*` for algorithm communication analysis.
4. Use `transport_*` for real serialized network cost.
5. For EGA, download-side reconstruction context is already counted in `parameter_download_*`, not only in `transport_*`.
