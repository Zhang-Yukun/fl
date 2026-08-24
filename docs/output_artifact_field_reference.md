# Output Artifacts and Config Field Reference

This document covers three things:

1. Runtime configuration fields and their meanings.
2. Output artifact schemas written by the framework.
3. wandb scalar and visualization key namespaces.

This reference matches the current implementation and does not document removed compatibility aliases.

## 1. Naming Conventions

### 1.1 Parameter communication vs transport communication

- `parameter_*`: logical algorithm-level communication volume. This counts the bytes strictly required to reconstruct the semantic payload at the receiver, including quantization scales, sparse indices, codebook/context payloads, and other algorithm-required metadata.
- `transport_*`: actual serialized bytes sent through the local or grpc transport, including serialization and protocol overhead.
- `*_compression_ratio` / `*_communication_ratio`: numerator is the dense FedAvg reference volume for the same round, denominator is the current method volume. Larger means better compression.

### 1.2 Protocol vs oracle evaluation

- `protocol_*`: metrics computed from the actually transmitted and actually reconstructed state.
- `oracle_*`: evaluation-only metrics computed from the exact uncompressed updates. Oracle never affects the real training loop.
- `active_*`: the evaluation path actually used for early stopping, best-round selection, and final `test`, controlled by `evaluation.mode`.

### 1.3 Attack metric names

- `primary_metric_name`: the attack metric currently used as the main privacy indicator.
- `primary_metric_value`: the value of that main metric.
- `objective_mse`: the optimization objective matching reconstructed gradients/updates to the intercepted target; it is not the same thing as input reconstruction MSE.
- `nearest_client_train_mse`: MSE between the reconstruction and the nearest sample in the attacked client's full local training set.
- `exact_target_mse`: MSE between the reconstruction and the exact attacked sample. For `update_payload`, this is opt-in and not emitted by default.

## 2. Configuration Fields

### 2.1 Common sections

#### `experiment`

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Experiment name. |
| `mode` | string | `centralized` or `federated`. |
| `output_dir` | string | Output directory. |

#### `runtime`

| Field | Type | Meaning |
| --- | --- | --- |
| `device` | string | Training device and default attack device, for example `cpu` or `cuda:0`. |
| `log_level` | string | loguru logging level. |
| `deterministic` | bool | Whether torch deterministic settings are enabled. |
| `seed` | int, optional | Global random seed. |

#### `task`

| Field | Type | Meaning |
| --- | --- | --- |
| `type` | string | Task type. Current time-series runs use `forecasting`. |

#### `data`

| Field | Type | Meaning |
| --- | --- | --- |
| `csv_path` | string | Raw merged CSV path. Some dataset builders override it with split client/server CSVs. |
| `clients` | list[string] | Client IDs. |
| `seq_len` | int | Input window length. |
| `pred_len` | int | Prediction horizon. |
| `batch_size` | int | DataLoader batch size. |
| `shuffle_train` | bool | Whether training loaders shuffle. Disabling it helps deterministic reruns. |
| `train_ratio` | float | Automatic train split ratio. |
| `val_ratio` | float | Automatic validation split ratio. |
| `num_workers` | int | DataLoader worker count. |

#### `model`

This block is interpreted by the concrete model implementation. Common fields in current forecasting models include:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Model name, for example `patchtst`. |
| `channels` | int | Input channel count. |
| `hidden_size` | int | Hidden width, when the selected model uses it. |
| `dropout` | float | Dropout probability. |

#### `training`

| Field | Type | Meaning |
| --- | --- | --- |
| `lr` | float | Training learning rate. |
| `patience` | int | Early-stop patience. Setting it to the total round count effectively disables early stop. |
| `min_delta` | float | Minimum validation improvement for early stop. |
| `loss` | string | Training loss. Supported values are `mse`, `mae`, `smooth_l1`, `huber`, and `task_default`. |
| `smooth_l1_beta` | float | `smooth_l1` beta. |
| `huber_delta` | float | `huber` delta. |
| `optimizer` | string | Training optimizer. Supported values are `adam`, `adamw`, and `sgd`. |
| `optimizer_eps` | float | `adam` / `adamw` epsilon. |
| `weight_decay` | float | Weight decay. |
| `momentum` | float | `sgd` momentum. |
| `nesterov` | bool | `sgd` Nesterov switch. |

#### `evaluation`

| Field | Type | Meaning |
| --- | --- | --- |
| `mode` | string | `protocol` or `oracle`. Controls best-validation selection and the final `test` payload. |
| `metrics` | list[string] | Extra evaluation metrics. The framework always keeps `mse`, `mae`, and `mape` available. |

#### `centralized`

| Field | Type | Meaning |
| --- | --- | --- |
| `rounds` | int | Total centralized training rounds. One round iterates over all train loaders once. |

#### `federated`

| Field | Type | Meaning |
| --- | --- | --- |
| `algorithm` | string | Federated method name. |
| `rounds` | int | Communication rounds. |
| `local_epochs` | int | Local epochs per communication round. |
| `local_steps` | int, optional | Optional local-step override. |
| `topk_fraction` | float, optional | Keep ratio for Top-k / Random-k / DP-Top-k style sparse methods. |
| `randomk_seed` | int, optional | Random seed for Random-k style methods. |
| `quantization_dtype` | string, optional | Quantization dtype such as `float16` or `qint8`. |
| `quantization_stochastic_rounding` | bool, optional | Whether stochastic rounding is used. |
| `quantization_seed` | int, optional | Quantization random seed. |
| `qsgd_levels` | int, optional | Number of QSGD quantization levels. |

Currently integrated method names are `fedavg`, `compressed_fedavg`, `sparse_fedavg`, `dp_topk_fedavg`, `randomk_fedavg`, `soteriafl`, `secure_quantized_fedavg`, `qsgd_fedavg`, `sign_fedavg`, `fedaware`, `adaptive_clipped_rdp_fedavg`, and `ega_fedavg`.

#### `transport`

| Field | Type | Meaning |
| --- | --- | --- |
| `upload_mode` | string | Client upload semantic, usually `update`, also supports `model`. |
| `download_mode` | string | Server download semantic, usually `model`, also supports `update`. |

#### `attack`

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | bool | Enable or disable attacks. |
| `target_type` | string | Attack target type: `gradient` or `update_payload`. |
| `reference_metric` | string | Primary attack metric: `reconstruction_mse` or `nearest_client_train_mse`. |
| `report_metrics` | list[string] | Extra attack metrics to expose. By default, `update_payload` only reports `nearest_client_train_mse`. |
| `steps` | int | Optimization steps per attack run. |
| `lr` | float | Attack learning rate. |
| `optimizer` | string | Attack optimizer, currently `adam` or `lbfgs`. |
| `restarts` | int | Random restarts per attack run. |
| `lbfgs_history_size` | int | `lbfgs` history size. |
| `input_clip` | float, optional | Optional clamp bound for dummy inputs. |
| `target_clip` | float, optional | Optional clamp bound for dummy targets in DLG. |
| `tv_weight` | float, optional | Total-variation regularizer weight. |
| `seed` | int, optional | Attack seed. |
| `success_mse_threshold` | float | Per-attack MSE success threshold. |
| `success_ssim_threshold` | float, optional | Optional SSIM threshold used only when the primary metric is `reconstruction_mse`. |
| `success_rate_threshold` | float | Aggregate pass/fail threshold used in attack summaries. |
| `data_range` | float | Data range used by PSNR and SSIM. |
| `client_selection` | string | Attacked-client selection strategy: `all`, `first`, or `round_robin`. |
| `clients_per_round` | int | Number of attacked clients on each attacked round. |
| `frequency_rounds` | int | Attack frequency in communication rounds. |
| `sample_count` | int | Number of independent attack evaluations per attacked client on that round, each using a different `batch_index`. |
| `max_samples` | int | Number of samples jointly reconstructed in one attack evaluation. |
| `model_mode` | string | Attack model mode, `train` or `eval`. |
| `local_optimizer` | string | Assumed local optimizer for `update_payload` inversion. |
| `local_lr` | float | Assumed local learning rate for `update_payload` inversion. |
| `local_optimizer_eps` | float | Assumed Adam epsilon for `update_payload` inversion. |
| `async_enabled` | bool | Enable asynchronous attack execution. |
| `async_workers` | int | Number of async attack workers. |
| `async_max_pending_rounds` | int | Maximum queued attack rounds. |
| `device` | string | Attack device. `same` inherits `runtime.device`. |

#### `tracking`

| Field | Type | Meaning |
| --- | --- | --- |
| `enabled` | bool | Enable wandb. |
| `offline` | bool | Use wandb offline mode. |
| `project` | string | wandb project name. |
| `name` | string, optional | wandb run name. |
| `group` | string, optional | wandb group. |
| `job_type` | string, optional | wandb job type. |
| `tags` | list[string], optional | wandb tags. |

#### `grpc`

| Field | Type | Meaning |
| --- | --- | --- |
| `address` | string | grpc server bind address. |
| `server_address` | string | grpc client target address. |
| `poll_seconds` | float | Client poll interval. |
| `max_message_mb` | float | Maximum grpc message size. |

#### `artifacts`

| Field | Type | Meaning |
| --- | --- | --- |
| `config_formats` | list[string] | Startup config formats. Supported values are `yaml`, `json`, and `toml`. |
| `save_every_rounds` | int | Snapshot interval. `0` disables snapshots. |

### 2.2 Algorithm-specific config blocks

#### `privacy`

Used by DP / privacy-aware methods only. Common fields include:

| Field | Type | Meaning |
| --- | --- | --- |
| `clip_norm` | float | Update clipping norm. |
| `noise_multiplier` | float | Gaussian noise multiplier. |
| `delta` | float | DP delta. |

#### `adaptive_clipped_rdp`

| Field | Type | Meaning |
| --- | --- | --- |
| `initial_clip_norm` | float | Initial clipping threshold. |
| `target_quantile` | float | Target quantile for adaptive clipping. |
| `learning_rate` | float | Clip-threshold update rate. |
| `noise_multiplier` | float | Added noise strength. |
| `delta` | float | DP delta. |
| `seed` | int | Random seed. |

#### `fedaware`

| Field | Type | Meaning |
| --- | --- | --- |
| `beta` | float, optional | Server-side FedAware blending factor. |

#### `ega`

EGA has a larger dedicated config surface. Use the active `configs/*ega*.yaml` files as the authoritative source for the exact field set. Common fields cover encoding width, codebook size, quantization width, pretraining schedule, pretraining optimizer settings, and round-context options.

## 3. Output Directory Files

### 3.1 Common files

| File | Meaning |
| --- | --- |
| `config.yaml` / `config.json` / `config.toml` | Effective runtime config saved before training starts, with defaults materialized. |
| `summary.json` | Final run summary. |
| `metrics.json` | Per-round detailed records. |
| `run.log` | Text logs written by loguru. |
| `model.pt` / `model.pt` | Best-validation model used for final testing. |
| `oracle_model.pt` | Saved only for federated runs with oracle evaluation. |
| `attack_results.json` | One JSON record per DLG/iDLG attack. Empty or missing when attacks are disabled. |
| `attack_artifacts/` | Raw tensor artifacts for attack reconstruction outputs. |
| `snapshots/round_xxxx/` | Intermediate snapshots written when `artifacts.save_every_rounds > 0`. |
| `resume_state.pt` | Snapshot resume state, only inside snapshot directories. |

### 3.2 `summary.json` for centralized runs

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | Final test metrics containing `mse`, `mae`, and `mape`. |
| `rounds` | int | Executed rounds. |
| `total_time_seconds` | float | Total runtime. |
| `best_round` | int | Best validation round index. |
| `best_val_mse` | float | Best validation MSE. |
| `best_val_mae` | float | Best validation MAE. |
| `best_val_mape` | float | Best validation MAPE. |
| `test_checkpoint` | string | Currently always `best_validation`. |

### 3.3 `metrics.json` for centralized runs

| Field | Type | Meaning |
| --- | --- | --- |
| `history` | list[object] | Per-round records. |
| `history[].round` | int | Round index. |
| `history[].train_loss` | float | Average training loss for that round. |
| `history[].val_mse` | float | Validation MSE. |
| `history[].val_mae` | float | Validation MAE. |
| `history[].val_mape` | float | Validation MAPE. |
| `history[].round_time_seconds` | float | Round runtime. |
| `history[].elapsed_time_seconds` | float | Elapsed runtime. |
| `test` | object | Final test metrics. |
| `rounds` | int | Executed rounds. |
| `total_time_seconds` | float | Total runtime. |
| `best_round` | int | Best validation round. |
| `best_val` | object | Best validation metric object. |
| `test_checkpoint` | string | Currently always `best_validation`. |

### 3.4 `summary.json` for federated runs

| Field | Type | Meaning |
| --- | --- | --- |
| `test` | object | Final test metrics for the active evaluation mode. |
| `active_test_scope` | string | Whether `test` comes from `protocol` or `oracle`. |
| `evaluation_mode` | string | Configured evaluation mode. |
| `best_val_scope` | string | Validation path used for best-round selection. |
| `protocol_test` | object | Protocol-path test metrics. |
| `oracle_test` | object | Oracle-path test metrics. When oracle is disabled, this equals `protocol_test`. |
| `rounds` | int | Executed communication rounds. |
| `total_time_seconds` | float | Total runtime. |
| `best_round` | int | Best validation round index. |
| `best_val_mse` | float | Best validation MSE. |
| `best_val_mae` | float | Best validation MAE. |
| `best_val_mape` | float | Best validation MAPE. |
| `test_checkpoint` | string | Currently always `best_validation`. |
| `last_parameter_upload_compression_ratio` | float | Last-round parameter upload compression ratio. |
| `last_parameter_total_communication_ratio` | float | Last-round parameter total communication ratio. |
| `last_parameter_upload_bytes` | int | Last-round parameter upload bytes. |
| `last_parameter_download_bytes` | int | Last-round parameter download bytes. |
| `last_parameter_total_bytes` | int | Last-round total parameter bytes. |
| `last_transport_upload_bytes` | int | Last-round actual upload bytes. |
| `last_transport_download_bytes` | int | Last-round actual download bytes. |
| `last_transport_total_bytes` | int | Last-round total actual transport bytes. |
| `last_transport_upload_overhead_bytes` | int | Last-round upload protocol/serialization overhead. |
| `last_transport_download_overhead_bytes` | int | Last-round download protocol/serialization overhead. |
| `last_transport_upload_compression_ratio` | float | Last-round actual upload compression ratio. |
| `last_transport_total_communication_ratio` | float | Last-round actual total communication ratio. |
| `total_parameter_upload_bytes` | int | Cumulative parameter upload bytes. |
| `total_parameter_download_bytes` | int | Cumulative parameter download bytes. |
| `total_parameter_bytes` | int | Cumulative total parameter bytes. |
| `total_transport_upload_bytes` | int | Cumulative actual upload bytes. |
| `total_transport_download_bytes` | int | Cumulative actual download bytes. |
| `total_transport_bytes` | int | Cumulative actual total bytes. |
| `total_transport_upload_overhead_bytes` | int | Cumulative upload overhead bytes. |
| `total_transport_download_overhead_bytes` | int | Cumulative download overhead bytes. |
| `attack_target_type` | string | Attack target type. |
| `attack_primary_metric_name` | string | Primary metric used by the attack summary. |
| `attack_primary_metric_direction` | string | Currently always `higher_is_more_private`. |
| `attack_overall_avg_primary_metric_value` | float or null | Average primary attack metric across all attack records. |
| `attack_overall_best_primary_metric_value` | float or null | Best primary attack metric across all attack records. For MSE-like metrics, this is the minimum. |
| `attack_success_rate` | float | Aggregate attack success rate. |
| `attack_evaluations` | int | Number of attack records. |
| `attack_summary` | object | Nested attack summary, described below. |
| `privacy_accountant` | string or null | Privacy accountant name. |
| `privacy_epsilon` | float or null | Current cumulative epsilon. |
| `privacy_delta` | float or null | Current delta. |
| `privacy_rdp_alpha` | float or null | Current best alpha. |
| `privacy_rdp_total` | float or null | Current cumulative RDP. |
| `privacy_sampling_rate` | float or null | Current sampling rate. |
| `adaptive_clip_norm` | float or null | Adaptive clipping threshold. |
| `adaptive_clip_median_norm` | float or null | Median client update norm. |
| `adaptive_reference_clip_norm` | float or null | Reference clip threshold. |
| `adaptive_noise_std` | float or null | Added noise standard deviation. |
| `privacy_trust_model` | string or null | Privacy trust model description. |
| `transport` | string, optional | Transport implementation label, for example `grpc`. |

### 3.5 `metrics.json` for federated runs

`metrics.json` is a list of `RoundRecord` objects.

#### Round-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `round` | int | Communication round index. |
| `algorithm` | string | Method name. |
| `train_loss` | float | Average client training loss for that round. |
| `val_mse` / `val_mae` / `val_mape` | float | Validation metrics for the current `active_val_scope`. |
| `round_time_seconds` | float | Round runtime. |
| `elapsed_time_seconds` | float | Elapsed runtime. |
| `model_parameters` | int | Global model parameter count. |
| `model_bytes` | int | Dense model byte size. |
| `total_download_bytes` / `total_upload_bytes` | int | Logical download / upload bytes. |
| `total_download_parameters` / `total_upload_parameters` | int | Logical download / upload parameter counts. |
| `total_parameter_download_bytes` / `total_parameter_upload_bytes` / `total_parameter_bytes` | int | Parameter communication volume. |
| `total_parameter_download_parameters` / `total_parameter_upload_parameters` | int | Parameter communication counts. |
| `total_transport_download_bytes` / `total_transport_upload_bytes` / `total_transport_bytes` | int | Actual transport bytes. |
| `total_transport_download_overhead_bytes` / `total_transport_upload_overhead_bytes` | int | Protocol and serialization overhead. |
| `fedavg_reference_upload_bytes` / `fedavg_reference_total_bytes` | int | Same-round dense FedAvg reference used for ratios. |
| `fedavg_reference_upload_parameters` | int | Same-round dense FedAvg upload parameter count. |
| `parameter_upload_compression_ratio` | float | `fedavg_reference_upload_bytes / total_parameter_upload_bytes`. |
| `parameter_total_communication_ratio` | float | `fedavg_reference_total_bytes / total_parameter_bytes`. |
| `transport_upload_compression_ratio` | float | `fedavg_reference_upload_bytes / total_transport_upload_bytes`. |
| `transport_total_communication_ratio` | float | `fedavg_reference_total_bytes / total_transport_bytes`. |
| `privacy_*` and `adaptive_*` | number/string/null | Same meanings as in `summary.json`, but recorded for the current round. |
| `evaluation_mode` | string | Configured evaluation mode. |
| `active_val_scope` | string | Validation path actually used for early-stop and best-round logic. |
| `active_val_mse` / `active_val_mae` / `active_val_mape` | float | Active validation metrics. |
| `protocol_val_mse` / `protocol_val_mae` / `protocol_val_mape` | float | Protocol-path validation metrics. |
| `oracle_val_mse` / `oracle_val_mae` / `oracle_val_mape` | float or null | Oracle-path validation metrics. |
| `clients` | list[object] | Per-client communication records. |

#### Client-level fields in `clients[]`

| Field | Type | Meaning |
| --- | --- | --- |
| `client_id` | string | Client ID. |
| `num_samples` | int | Number of local training samples used this round. |
| `loss` | float | Local training loss. |
| `aggregation_payload_kind` | string | Semantic payload type sent into aggregation. |
| `evaluation_payload_kind` | string | Payload type used for server-side evaluation. |
| `download_bytes` / `upload_bytes` | int | Logical download / upload bytes for this client. |
| `download_parameters` / `upload_parameters` | int | Logical download / upload parameter counts. |
| `parameter_download_bytes` / `parameter_upload_bytes` | int | Parameter communication bytes for this client. |
| `parameter_download_parameters` / `parameter_upload_parameters` | int | Parameter communication counts for this client. |
| `dense_download_reference_bytes` / `dense_upload_reference_bytes` | int | Dense reference bytes. |
| `dense_download_reference_parameters` / `dense_upload_reference_parameters` | int | Dense reference parameter counts. |
| `transport_download_bytes` / `transport_upload_bytes` | int | Actual transport bytes. |
| `transport_download_overhead_bytes` / `transport_upload_overhead_bytes` | int | Transport overhead bytes. |
| `compressor` | string | Effective compressor name. |
| `privacy_clip_norm` | float | Client clipping threshold. |
| `privacy_noise_multiplier` | float | Client noise multiplier. |
| `aggregation_weight` | float | Aggregation weight. |

### 3.6 `attack_results.json`

Each element is one DLG or iDLG result.

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | string | Attack name, usually `DLG` or `iDLG`. |
| `primary_metric_name` | string | Primary attack metric name. |
| `primary_metric_value` | float or null | Primary attack metric value. |
| `psnr` | float or null | PSNR against the exact attacked sample. |
| `ssim` | float or null | SSIM against the exact attacked sample. |
| `iterations` | int | Attack optimization steps. |
| `time_seconds` | float | Attack runtime. |
| `success` | bool | Whether this attack passed the configured success rule. |
| `success_threshold` | float | Per-attack success threshold. |
| `objective_mse` | float | Optimization objective value. |
| `target_type` | string | `gradient` or `update_payload`. |
| `exact_target_mse` | float, optional | MSE against the exact attacked sample. |
| `nearest_client_train_mse` | float, optional | MSE against the nearest sample in the attacked client's training set. |
| `nearest_client_train_indices` | list[int], optional | Indices of those nearest training samples. |
| `client_id` | string | Attacked client ID. |
| `round_index` | int | Communication round index. |
| `sample_index` | int | Independent attack-evaluation index for that client and round. |
| `artifact_path` | string | Relative path to the `.pt` tensor artifact. |
| `reference_label` | string | Visualization reference source: `exact_target` or `nearest_client_train`. |

### 3.7 Nested `attack_summary`

`summary.json.attack_summary` contains:

| Field | Type | Meaning |
| --- | --- | --- |
| `target_type` | string | Attack target type. |
| `primary_metric_name` | string | Primary metric name. |
| `primary_metric_direction` | string | Currently always `higher_is_more_private`. |
| `success_rate_threshold` | float | Aggregate pass/fail threshold. |
| `overall_avg_primary_metric_value` | float or null | Average primary metric across all attack results. |
| `overall_best_primary_metric_value` | float or null | Best primary metric across all attack results. |
| `overall_success_rate` | float | Aggregate attack success rate. |
| `overall_passes` | bool | Whether the aggregate success rate is at most the threshold. |
| `methods` | object | Method-level attack summaries. |
| `clients` | object | Client-level attack summaries. |

`methods[<method>]`, `clients[<client_id>]`, and `clients[<client_id>].methods[<method>]` all use this schema:

| Field | Type | Meaning |
| --- | --- | --- |
| `primary_metric_name` | string | Primary metric name for that subset. |
| `count` | int | Number of attack results in that subset. |
| `success_rate` | float | Success rate for that subset. |
| `passes` | bool | Whether that subset passes the threshold. |
| `avg_primary_metric_value` | float or null | Average primary metric for that subset. |
| `best_primary_metric_value` | float or null | Best primary metric for that subset. |

## 4. wandb Logged Fields

### 4.1 Common scalars

| Key | Meaning |
| --- | --- |
| `tracking/step` | Explicit logging step. This keeps async attack logging aligned. |
| `run/model_parameters` | Model parameter count. |
| `run/model_bytes` | Dense model byte size. |
| `run/mode` | `centralized` or `federated`. |
| `run/elapsed_time_seconds` | Current elapsed runtime. |
| `run/total_time_seconds` | Final total runtime. |
| `run/best_round` | Best validation round. |
| `run/best_val_mse` / `run/best_val_mae` / `run/best_val_mape` | Best validation metrics. |
| `test/mse` / `test/mae` / `test/mape` | Final centralized test metrics. |
| `protocol_test/<metric>` | Final federated protocol-path test metrics. |
| `oracle_test/<metric>` | Final federated oracle-path test metrics. |

### 4.2 Centralized per-round scalars

| Key | Meaning |
| --- | --- |
| `round/loss` | Round training loss. |
| `round/val_mse` / `round/val_mae` / `round/val_mape` | Validation metrics. |
| `round/time_seconds` | Round runtime. |

### 4.3 Federated per-round scalars

#### Round-level keys

wandb logs every `RoundRecord` field as `round/<field>`, including:

- `round/round`
- `round/algorithm`
- `round/train_loss`
- `round/val_mse`, `round/val_mae`, `round/val_mape`
- `round/round_time_seconds`, `round/elapsed_time_seconds`
- `round/model_parameters`, `round/model_bytes`
- `round/total_download_bytes`, `round/total_upload_bytes`
- `round/total_download_parameters`, `round/total_upload_parameters`
- `round/total_parameter_download_bytes`, `round/total_parameter_upload_bytes`, `round/total_parameter_bytes`
- `round/total_parameter_download_parameters`, `round/total_parameter_upload_parameters`
- `round/total_transport_download_bytes`, `round/total_transport_upload_bytes`, `round/total_transport_bytes`
- `round/total_transport_download_overhead_bytes`, `round/total_transport_upload_overhead_bytes`
- `round/fedavg_reference_upload_bytes`, `round/fedavg_reference_upload_parameters`, `round/fedavg_reference_total_bytes`
- `round/parameter_upload_compression_ratio`, `round/parameter_total_communication_ratio`
- `round/transport_upload_compression_ratio`, `round/transport_total_communication_ratio`
- `round/privacy_*`, `round/adaptive_*`
- `round/evaluation_mode`, `round/active_val_scope`
- `round/active_val_mse`, `round/active_val_mae`, `round/active_val_mape`
- `round/protocol_val_mse`, `round/protocol_val_mae`, `round/protocol_val_mape`
- `round/oracle_val_mse`, `round/oracle_val_mae`, `round/oracle_val_mape`

#### Client-level keys

Each client record is logged as `client/<client_id>/<ClientCommunicationRecord field>`, for example:

- `client/Nd2O3/loss`
- `client/Nd2O3/aggregation_payload_kind`
- `client/Nd2O3/parameter_upload_bytes`
- `client/Nd2O3/transport_upload_bytes`
- `client/Nd2O3/compressor`
- `client/Nd2O3/aggregation_weight`

#### Cumulative communication keys

| Key | Meaning |
| --- | --- |
| `cumulative/last_parameter_upload_bytes` and peers | Snapshot of the latest round values. |
| `cumulative/total_parameter_upload_bytes` / `download` / `bytes` | Cumulative parameter communication volume. |
| `cumulative/total_transport_upload_bytes` / `download` / `bytes` | Cumulative actual transport volume. |
| `cumulative/total_transport_upload_overhead_bytes` / `download` | Cumulative protocol overhead. |
| `cumulative/last_parameter_upload_compression_ratio` | Latest-round parameter upload compression ratio. |
| `cumulative/last_parameter_total_communication_ratio` | Latest-round parameter total communication ratio. |
| `cumulative/last_transport_upload_compression_ratio` | Latest-round actual upload compression ratio. |
| `cumulative/last_transport_total_communication_ratio` | Latest-round actual total communication ratio. |

### 4.4 Attack scalar keys

#### Global attack keys

| Key | Meaning |
| --- | --- |
| `attack/round_index` | Round index for this attack payload. |
| `attack/time_seconds` | Total time spent by all attack evaluations for that round. |
| `attack/evaluations_this_round` | Number of attack records produced on that round. |
| `attack/clients_this_round` | Number of attacked clients on that round. |
| `attack/evaluations_per_client_this_round` | Number of independent attack evaluations per attacked client on that round. |
| `attack/primary_metric_name` | Current primary attack metric name. |
| `attack/cumulative_avg_primary_metric_value` | Average primary metric over all attack results seen so far. |
| `attack/cumulative_success_rate` | Success rate over all attack results seen so far. |

#### Method-level attack keys

The prefix is `attack/<method>/`, for example `attack/DLG/` or `attack/iDLG/`:

| Key suffix | Meaning |
| --- | --- |
| `primary_metric_name` | Primary metric name for that method. |
| `primary_metric_value` | Current-round average primary metric for that method. |
| `cumulative_avg_primary_metric_value` | Running average primary metric for that method. |
| `exact_target_mse` | Current-round average exact-target MSE. |
| `cumulative_avg_exact_target_mse` | Running average exact-target MSE. |
| `nearest_client_train_mse` | Current-round average nearest-train MSE. |
| `cumulative_avg_nearest_client_train_mse` | Running average nearest-train MSE. |
| `psnr` / `ssim` | Current-round average PSNR / SSIM. |
| `iterations` | Attack optimization steps. |
| `time_seconds` | Current-round average attack runtime. |
| `objective_mse` | Current-round average optimization objective. |
| `cumulative_avg_objective_mse` | Running average optimization objective. |
| `success_fraction` | Current-round success rate. |
| `cumulative_success_rate` | Running success rate. |

#### Client-level attack keys

The prefix is `attack/client/<client_id>/`, with the same suffix set as the method-level keys. There is also `attack/client/<client_id>/<method>/...` for one method under one client.

## 5. wandb Visualization Keys

### 5.1 Prediction plots

| Key | Meaning |
| --- | --- |
| `prediction/centralized/val` | First validation-batch, first-sample input/prediction/target plot for centralized training. |
| `prediction/centralized/test` | Centralized test plot. |
| `prediction/federated/val_protocol` | Federated protocol-path validation plot. |
| `prediction/federated/val_oracle` | Federated oracle-path validation plot. |
| `prediction/federated/test_protocol` | Federated protocol-path test plot. |
| `prediction/federated/test_oracle` | Federated oracle-path test plot. |
| `prediction/grpc/val_protocol` / `val_oracle` / `test_protocol` / `test_oracle` | grpc-mode counterparts. |

These plots are inverse-transformed back to the original data scale by default and contain the input context, the predicted horizon, and the true target horizon in the same figure.

### 5.2 Attack reconstruction plots

| Key | Meaning |
| --- | --- |
| `attack/<method>/reconstruction` | First reconstruction figure for that method on a round. |
| `attack/client/<client_id>/<method>/reconstruction` | First reconstruction figure for that client and method on a round. |

Notes:

- The reference curve in the figure follows the same semantics as the recorded primary attack metric.
- When the primary metric is `nearest_client_train_mse`, the plotted reference is the nearest sample from the attacked client's training set, not the exact attacked batch.
- When the primary metric is `reconstruction_mse`, the plotted reference is the exact attacked sample.
