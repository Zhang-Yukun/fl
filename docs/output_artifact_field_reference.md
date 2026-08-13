# Output Artifact Field Reference

This document explains the training result files written under one experiment output directory, such as `outputs/<run_name>/`.

It covers three execution modes:
- centralized training
- single-process federated training
- gRPC multi-process federated training

## Output Directory Overview

A typical run directory may contain these files:

- `summary.json`: final run summary, intended for quick comparison
- `metrics.json`: detailed training history
- `attack_results.json`: per-attack evaluation records, only when attack evaluation is enabled
- `config.yaml` / `config.json` / `config.toml`: saved experiment configuration
- `model.pt`: final federated global model state
- `centralized_model.pt`: final centralized model state
- `run.log`: loguru runtime log when logging is enabled
- `client_<client_id>/` or `client_<client_id>.log`: client-side logs for gRPC runs

## File-Level Meaning

| File | Meaning | Modes |
| --- | --- | --- |
| `summary.json` | Final compact summary of performance, communication, and attack statistics | centralized / federated / gRPC |
| `metrics.json` | Full training history. Centralized mode stores epoch history; federated mode stores round history | centralized / federated / gRPC |
| `attack_results.json` | One record per DLG or iDLG evaluation | federated / gRPC when `attack.enabled=true` |
| `config.*` | Saved runtime configuration after include resolution and CLI overrides | all |
| `model.pt` | Final federated global model parameters | federated / gRPC |
| `centralized_model.pt` | Final centralized model parameters | centralized |

## `summary.json`

### Centralized Mode

Centralized `summary.json` contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | Final test metrics on the server test loader |
| `test.mse` | float | Mean squared error on the test set |
| `test.mae` | float | Mean absolute error on the test set |
| `test.mape` | float | Mean absolute percentage error on the test set |
| `epochs` | int | Number of completed training epochs |
| `total_time_seconds` | float | Total wall-clock training time |

### Federated and gRPC Modes

Federated `summary.json` is the main artifact for final comparison.

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | Final test metrics after the last global model is produced |
| `test.mse` | float | Mean squared error on the server test set |
| `test.mae` | float | Mean absolute error on the server test set |
| `test.mape` | float | Mean absolute percentage error on the server test set |
| `rounds` | int | Number of completed federated rounds actually recorded |
| `total_time_seconds` | float | Total end-to-end wall-clock time |
| `last_upload_compression_ratio` | float | FedAvg reference upload bytes divided by the actual parameter upload bytes of the last round |
| `last_total_communication_ratio` | float | FedAvg reference total bytes divided by the actual parameter total bytes of the last round |
| `last_parameter_upload_bytes` | int | Parameter bytes uploaded by all clients in the last round |
| `last_parameter_download_bytes` | int | Parameter bytes downloaded by all clients in the last round |
| `last_parameter_total_bytes` | int | Sum of last-round parameter upload and parameter download bytes |
| `last_transport_upload_bytes` | int | Serialized transport upload bytes in the last round, including protocol/payload packaging |
| `last_transport_download_bytes` | int | Serialized transport download bytes in the last round |
| `last_transport_total_bytes` | int | Sum of last-round transport upload and transport download bytes |
| `last_transport_upload_overhead_bytes` | int | Extra upload bytes beyond pure model parameter payload in the last round |
| `last_transport_download_overhead_bytes` | int | Extra download bytes beyond pure model parameter payload in the last round |
| `last_transport_upload_compression_ratio` | float | FedAvg reference upload bytes divided by actual transport upload bytes in the last round |
| `last_transport_total_communication_ratio` | float | FedAvg reference total bytes divided by actual transport total bytes in the last round |
| `total_parameter_upload_bytes` | int | Cumulative parameter upload bytes across all completed rounds |
| `total_parameter_download_bytes` | int | Cumulative parameter download bytes across all completed rounds |
| `total_parameter_bytes` | int | Cumulative parameter upload + download bytes across all rounds |
| `total_transport_upload_bytes` | int | Cumulative serialized transport upload bytes across all rounds |
| `total_transport_download_bytes` | int | Cumulative serialized transport download bytes across all rounds |
| `total_transport_bytes` | int | Cumulative serialized transport bytes across all rounds |
| `total_transport_upload_overhead_bytes` | int | Cumulative upload overhead bytes beyond pure parameter payload |
| `total_transport_download_overhead_bytes` | int | Cumulative download overhead bytes beyond pure parameter payload |
| `attack_target_type` | string | Target intercepted by the attack, usually `gradient` or `update_payload` |
| `attack_primary_metric` | string | Metric used as the main privacy indicator |
| `attack_primary_metric_direction` | string | Interpretation direction of the privacy metric, currently `higher_is_more_private` |
| `attack_overall_avg_mse` | float or null | Average value of the primary attack metric over all attack records |
| `attack_success_rate` | float | Fraction of attack records counted as successful |
| `attack_evaluations` | int | Total number of attack records written to `attack_results.json` |
| `attack_summary` | object | Aggregated privacy summary grouped by attack method |
| `transport` | string | Present only in gRPC mode; currently fixed to `grpc` |

### `attack_summary` in `summary.json`

`attack_summary` is produced from all attack records.

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_metric` | string | Main privacy metric used in this run |
| `primary_metric_direction` | string | How to interpret the metric; larger means harder to reconstruct |
| `target_type` | string or null | `gradient` or `update_payload` |
| `success_rate_threshold` | float | Threshold used to decide whether the method passes the privacy criterion |
| `overall_avg_mse` | float or null | Mean of the primary metric over all attack records |
| `overall_best_mse` | float or null | Best (smallest) primary metric value observed |
| `overall_avg_exact_target_mse` | float or null | Average MSE to the exact attacked batch/window |
| `overall_avg_nearest_client_train_mse` | float or null | Average MSE to the nearest sample in the attacked client training set |
| `overall_avg_psnr` | float or null | Mean PSNR over all attack records |
| `overall_avg_ssim` | float or null | Mean SSIM over all attack records |
| `overall_avg_gradient_mse` | float or null | Mean attack objective mismatch in gradient/update space |
| `overall_avg_objective_mse` | float or null | Alias of `overall_avg_gradient_mse` |
| `overall_success_rate` | float | Fraction of successful attacks across all records |
| `overall_success_rate_percent` | float | Success rate in percentage form |
| `overall_passes` | bool | Whether overall attack success rate is not higher than `success_rate_threshold` |
| `methods` | object | Per-method aggregation, keyed by attack name such as `DLG` or `iDLG` |

Each `attack_summary.methods.<method_name>` entry contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_metric` | string | Primary metric used for this method |
| `target_type` | string or null | Intercepted target type for this method |
| `success_count` | int | Number of successful attacks for this method |
| `total_count` | int | Number of attack records for this method |
| `success_rate` | float | `success_count / total_count` |
| `success_rate_percent` | float | Percentage version of `success_rate` |
| `avg_mse` | float or null | Mean primary metric value for this method |
| `best_mse` | float or null | Best primary metric value for this method |
| `avg_exact_target_mse` | float or null | Mean exact-batch reconstruction MSE |
| `avg_nearest_client_train_mse` | float or null | Mean nearest-train-window reconstruction MSE |
| `avg_psnr` | float or null | Mean PSNR |
| `avg_ssim` | float or null | Mean SSIM |
| `best_ssim` | float or null | Best SSIM observed |
| `avg_gradient_mse` | float or null | Mean attack objective mismatch |
| `avg_objective_mse` | float or null | Alias of `avg_gradient_mse` |
| `avg_time_seconds` | float or null | Mean optimization time per attack |
| `passes` | bool | Whether this method's success rate is not higher than `success_rate_threshold` |

## `metrics.json`

### Centralized Mode

Centralized `metrics.json` is an object:

| Field | Type | Meaning |
| --- | --- | --- |
| `history` | list[object] | Per-epoch training history |
| `test` | object | Final test metrics |
| `epochs` | int | Number of recorded epochs |
| `total_time_seconds` | float | Total wall-clock time |

Each item of `history` contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `epoch` | int | Epoch index starting from 0 |
| `train_loss` | float | Mean training loss across all client loaders treated centrally |
| `val_mse` | float | Validation MSE |
| `val_mae` | float | Validation MAE |
| `val_mape` | float | Validation MAPE |
| `epoch_time_seconds` | float | Wall-clock time of this epoch |
| `elapsed_time_seconds` | float | Elapsed wall-clock time since training started |

### Federated and gRPC Modes

Federated `metrics.json` is a list of round records. Each item corresponds to one `RoundRecord`.

#### Round-Level Fields

| Field | Type | Meaning |
| --- | --- | --- |
| `round` | int | Round index starting from 0 |
| `algorithm` | string | Federated algorithm name from config, e.g. `fedavg`, `secure_quantized_fedavg` |
| `train_loss` | float | Mean of client local training losses in this round |
| `val_mse` | float | Global validation MSE after aggregation |
| `val_mae` | float | Global validation MAE after aggregation |
| `val_mape` | float | Global validation MAPE after aggregation |
| `round_time_seconds` | float | Wall-clock time spent on this round |
| `elapsed_time_seconds` | float | Elapsed wall-clock time since run start |
| `model_parameters` | int | Number of parameters in the current global model |
| `model_bytes` | int | Serialized byte size of the current global model |
| `total_download_bytes` | int | Last-round total parameter download bytes, equal to `total_parameter_download_bytes` |
| `total_download_parameters` | int | Last-round total downloaded parameter count |
| `total_upload_bytes` | int | Last-round total parameter upload bytes, equal to `total_parameter_upload_bytes` |
| `total_upload_parameters` | int | Last-round total uploaded parameter count |
| `total_parameter_download_bytes` | int | Parameter download bytes only, summed over all clients |
| `total_parameter_download_parameters` | int | Downloaded parameter count only, summed over all clients |
| `total_parameter_upload_bytes` | int | Parameter upload bytes only, summed over all clients |
| `total_parameter_upload_parameters` | int | Uploaded parameter count only, summed over all clients |
| `total_parameter_bytes` | int | `total_parameter_download_bytes + total_parameter_upload_bytes` |
| `total_transport_download_bytes` | int | Serialized transport download bytes summed over all clients |
| `total_transport_upload_bytes` | int | Serialized transport upload bytes summed over all clients |
| `total_transport_bytes` | int | Serialized transport upload + download bytes |
| `total_transport_download_overhead_bytes` | int | Extra download bytes beyond pure parameter payload |
| `total_transport_upload_overhead_bytes` | int | Extra upload bytes beyond pure parameter payload |
| `fedavg_reference_upload_bytes` | int | Dense FedAvg upload baseline in bytes for this round |
| `fedavg_reference_upload_parameters` | int | Dense FedAvg upload baseline in parameter count |
| `fedavg_reference_total_bytes` | int | Dense FedAvg upload + download baseline in bytes for this round |
| `upload_compression_ratio` | float | `fedavg_reference_upload_bytes / total_parameter_upload_bytes` |
| `total_communication_ratio` | float | `fedavg_reference_total_bytes / total_parameter_bytes` |
| `communication_ratio` | float | Alias of `upload_compression_ratio` |
| `transport_upload_compression_ratio` | float | `fedavg_reference_upload_bytes / total_transport_upload_bytes` |
| `transport_total_communication_ratio` | float | `fedavg_reference_total_bytes / total_transport_bytes` |
| `clients` | list[object] | Per-client communication records for this round |

#### Per-Client Record Fields in `metrics.json[*].clients[*]`

| Field | Type | Meaning |
| --- | --- | --- |
| `client_id` | string | Client name, e.g. `Nd2O3` |
| `num_samples` | int | Number of local training windows used to weight aggregation |
| `loss` | float | Mean local training loss returned by this client |
| `payload_kind` | string | Semantic type of transmitted payload |
| `download_bytes` | int | Parameter download bytes attributed to this client; equal to `parameter_download_bytes` |
| `download_parameters` | int | Downloaded parameter count attributed to this client |
| `parameter_download_bytes` | int | Pure model-parameter download bytes |
| `parameter_download_parameters` | int | Pure model-parameter count downloaded |
| `dense_download_reference_bytes` | int | Dense FedAvg reference download bytes for comparison |
| `dense_download_reference_parameters` | int | Dense FedAvg reference download parameter count |
| `upload_bytes` | int | Parameter upload bytes attributed to this client; equal to `parameter_upload_bytes` |
| `upload_parameters` | int | Uploaded parameter count attributed to this client |
| `parameter_upload_bytes` | int | Pure model-parameter upload bytes |
| `parameter_upload_parameters` | int | Pure model-parameter count uploaded |
| `transport_download_bytes` | int | Serialized transport bytes actually downloaded by this client |
| `transport_upload_bytes` | int | Serialized transport bytes actually uploaded by this client |
| `transport_download_overhead_bytes` | int | Download-side transport overhead beyond pure parameter payload |
| `transport_upload_overhead_bytes` | int | Upload-side transport overhead beyond pure parameter payload |
| `dense_upload_reference_bytes` | int | Dense FedAvg reference upload bytes for comparison |
| `dense_upload_reference_parameters` | int | Dense FedAvg reference upload parameter count |
| `compressor` | string | Compressor or payload encoding name |
| `privacy_clip_norm` | float | DP clipping norm applied before transmission; `0.0` means not used |
| `privacy_noise_multiplier` | float | DP noise multiplier applied before transmission; `0.0` means not used |
| `aggregation_weight` | float | Client weight used during server aggregation |

### Common `payload_kind` Values

| Value | Meaning |
| --- | --- |
| `dense_update` | Standard FedAvg dense model update |
| `quantized_update` | Dense update quantized before transmission |
| `sparse_update` | Top-k sparse update |
| `dp_topk_dp_update` | Top-k sparse update after clipping/noise |
| `soteriafl_randomk_dp_update` | Random-k sparse DP update |
| `fedpetuning_trainable_state` | Only trainable PEFT parameters are transmitted |

### Common `compressor` Values

| Value | Meaning |
| --- | --- |
| `none` | No compression |
| `topk` | Top-k sparsification |
| `topk_dp` | Top-k sparsification with DP privatization |
| `randomk_unbiased` | Random-k style unbiased sparse compressor |
| `float16_quantized_dense` | Dense update quantized to float16 |
| `int8_quantized_dense` | Dense update quantized to int8 |
| `trainable_subset` | Only trainable parameter subset is sent |

## `attack_results.json`

`attack_results.json` is a list. Each item corresponds to one `AttackResult` converted to JSON.

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Attack method name, usually `DLG` or `iDLG` |
| `mse` | float or null | Primary privacy metric value used for this record |
| `psnr` | float or null | Peak signal-to-noise ratio of the reconstruction |
| `ssim` | float or null | Structural similarity score |
| `iterations` | int | Number of optimizer steps used in the attack loop |
| `time_seconds` | float | Wall-clock time spent by this attack record |
| `success` | bool | Whether this attack is counted as successful under current thresholds |
| `success_threshold` | float | MSE threshold used for success decision |
| `gradient_mse` | float or null | Optimization objective mismatch in gradient/update space |
| `objective_mse` | float or null | Alias of `gradient_mse` |
| `target_type` | string | `gradient` or `update_payload` |
| `exact_target_mse` | float or null | MSE to the exact attacked batch/window |
| `nearest_client_train_mse` | float or null | MSE to the nearest sample in the attacked client's full training set |
| `nearest_client_train_indices` | list[int] or null | Indices of the nearest training windows used for scoring |
| `metric_name` | string | Metric used as the primary privacy indicator for this record |

### Meaning of `metric_name`

| Value | Meaning |
| --- | --- |
| `reconstruction_mse` | Privacy is judged by direct reconstruction error to the attacked batch |
| `nearest_client_train_mse` | Privacy is judged by the nearest reconstruction error to the attacked client's whole training set |

## `config.yaml`, `config.json`, `config.toml`

These files store the resolved experiment configuration after:
- YAML includes are expanded
- command-line overrides are applied

The default saved format is YAML. Additional formats are controlled by `artifacts.config_formats`.

These files are intended for:
- exact experiment reproduction
- checking runtime hyperparameters after override resolution
- comparing train/attack/communication settings across runs

## `model.pt` and `centralized_model.pt`

| File | Meaning |
| --- | --- |
| `model.pt` | Final federated global model state dictionary |
| `centralized_model.pt` | Final centralized training model state dictionary |

## Practical Notes

- In federated mode, prefer `total_parameter_*` fields when comparing model communication volume across algorithms.
- In gRPC mode, `total_transport_*` also includes protocol and control traffic, such as repeated polling for global state.
- `summary.json` is the best file for final comparison across runs.
- `metrics.json` is the best file for plotting convergence curves or per-round communication trends.
- `attack_results.json` is the best file for detailed privacy analysis and per-method inspection.
