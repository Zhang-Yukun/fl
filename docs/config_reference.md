# Framework Configuration Reference

本文档说明当前框架会读取到的配置块、默认值、常见可选值以及作用。默认值以 `fedlab/utils/config.py` 的 `_RUNTIME_DEFAULTS` 和当前注册表实现为准。

## 1. 加载与合并规则

### 1.1 `includes`

- 每个 YAML 文件都可以使用 `includes` 引入其他 YAML。
- `includes` 按顺序加载，后面的 include 会覆盖前面的同名叶子值。
- 当前文件最后再覆盖 `includes` 合并后的结果，所以当前文件优先级最高。
- 如果同一个键两边都是字典，会递归合并；如果一边不是字典，则直接覆盖。

示例：

```yaml
# a.yaml
A:
  a: 1

# b.yaml
A:
  b: 2

# main.yaml
includes:
  - a.yaml
  - b.yaml
```

最终得到：

```yaml
A:
  a: 1
  b: 2
```

### 1.2 CLI 覆盖

运行入口支持 `--override key=value`。

例如：

```bash
--override runtime.device=cuda:0
--override attack.enabled=false
--override federated.rounds=20
```

## 2. 顶层配置块

当前框架会读取这些顶层块：

- `experiment`
- `runtime`
- `task`
- `data`
- `model`
- `training`
- `evaluation`
- `centralized`
- `federated`
- `transport`
- `attack`
- `tracking`
- `grpc`
- `artifacts`
- `privacy`
- `adaptive_clipped_rdp`
- `ega`

## 3. 通用配置块

### 3.1 `experiment`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `experiment.name` | 无固定默认实验名 | 任意字符串 | 运行名，用于日志、输出目录、tracking。 |
| `experiment.mode` | `federated` | `federated`, `centralized` | 选择联邦训练还是集中训练入口。 |
| `experiment.output_dir` | 无固定默认目录 | 任意合法路径 | 本次运行的输出目录。 |

### 3.2 `runtime`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `runtime.device` | `cpu` | `cpu`, `cuda:0`, `cuda:1` 等 | 主训练设备；`attack.device=same` 时会继承。 |
| `runtime.log_level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` | 日志等级。 |
| `runtime.deterministic` | `true` | `true`, `false` | 是否启用确定性运行。 |
| `runtime.seed` | 无 | 任意整数 | 随机种子，训练和部分数据加载会使用。 |
| `runtime.num_threads` | 无 | 正整数 | PyTorch intra-op 线程数。 |
| `runtime.num_interop_threads` | 无 | 正整数 | PyTorch inter-op 线程数。 |

### 3.3 `task`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `task.type` | `forecasting` | `forecasting`, `classification` | 决定任务插件、默认损失、默认指标以及部分攻击逻辑。 |

### 3.4 `training`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `training.lr` | `0.001` | 正浮点数 | 训练优化器学习率。 |
| `training.patience` | `50` | 非负整数 | 早停耐心轮数。 |
| `training.min_delta` | `0.0` | 非负浮点数 | 早停的最小改进阈值。 |
| `training.loss` | 随任务决定；框架默认 `mse` | `mse`, `mae`, `smooth_l1`, `huber`, `cross_entropy`, `task_default` | 训练损失。分类通常用 `cross_entropy`。 |
| `training.optimizer` | `adam` | `adam`, `adamw`, `sgd` | 训练优化器。 |
| `training.optimizer_eps` | `1e-8` | 正浮点数 | `adam`/`adamw` 的 `eps`。 |
| `training.weight_decay` | `0.0` | 非负浮点数 | 权重衰减。 |
| `training.momentum` | `0.0` | 非负浮点数 | `sgd` 的动量。 |
| `training.nesterov` | `false` | `true`, `false` | 是否对 `sgd` 启用 Nesterov。 |
| `training.smooth_l1_beta` | `1.0` | 正浮点数 | `smooth_l1` 损失的 `beta`。 |
| `training.huber_delta` | `1.0` | 正浮点数 | `huber` 损失的 `delta`。 |

### 3.5 `evaluation`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `evaluation.metrics` | forecasting 默认 `[mse, mae, mape]` | 由任务注册表支持；当前常用 `mse`, `mae`, `mape`, `accuracy`, `cross_entropy` | 评测指标列表。任务主指标会被自动补全。 |

### 3.6 `centralized`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `centralized.rounds` | `10` | 正整数 | 集中训练轮数。 |

### 3.7 `federated`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `federated.algorithm` | `fedavg` | `fedavg`, `adaptive_clipped_rdp_fedavg`, `sparse_fedavg`, `randomk_fedavg`, `sign_fedavg`, `qsgd_fedavg`, `secure_quantized_fedavg`, `ega_fedavg` | 联邦算法名。 |
| `federated.rounds` | `20` | 正整数 | 联邦通信轮数。 |
| `federated.local_epochs` | `1` | 正整数 | 每轮本地训练 epoch 数。 |
| `federated.local_steps` | 无 | 正整数 | 若配置则优先于 `local_epochs`。 |
| `federated.topk_fraction` | 无 | `(0, 1]` 浮点数 | `sparse_fedavg` / `randomk_fedavg` 的保留比例。 |
| `federated.randomk_seed` | 无 | 整数 | `randomk_fedavg` 稀疏采样种子。 |
| `federated.qsgd_levels` | 无 | 正整数 | `qsgd_fedavg` 量化级数。 |
| `federated.quantization_dtype` | 无 | 常用 `float16`, `qint8` | `secure_quantized_fedavg` 量化类型。 |
| `federated.quantization_stochastic_rounding` | 无 | `true`, `false` | 是否对量化使用随机舍入。 |
| `federated.quantization_seed` | 无 | 整数 | 量化相关随机种子。 |

### 3.8 `transport`

`transport.upload_mode` 与 `transport.download_mode` 已移除配置面。框架固定采用：上传语义为 `update`，下载语义为 `model`。

### 3.9 `attack`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `attack.enabled` | `true` | `true`, `false` | 是否启用攻击。 |
| `attack.target_type` | `update_payload` | 当前仅支持 `update_payload` | 攻击目标类型。 |
| `attack.reference_metric` | `nearest_client_train_mse` | `auto`, `reconstruction_mse`, `nearest_client_train_mse` | 兼容旧字段，用于补充输出参考口径。 |
| `attack.report_metrics` | `[nearest_client_train_mse]` | `auto` 或包含 `exact_target_mse` / `nearest_client_train_mse` 的列表 | 额外输出哪些参考指标。 |
| `attack.steps` | `300` | 正整数 | 每次 DLG / iDLG 优化步数。 |
| `attack.lr` | `0.001` | 正浮点数 | 攻击优化学习率。 |
| `attack.optimizer` | `adam` | 当前仅 `adam` | 攻击优化器。 |
| `attack.restarts` | `1` | 正整数 | 随机重启次数。 |
| `attack.input_clip` | 无 | 浮点数 | 对 `dummy_x` 的裁剪上界。 |
| `attack.target_clip` | 无 | 浮点数 | 对 DLG `dummy_y` 的裁剪上界。 |
| `attack.tv_weight` | 无 | 非负浮点数 | 输入总变差正则权重。 |
| `attack.seed` | 无 | 整数 | 攻击随机种子。 |
| `attack.recovery_match_metric` | `mse` | `mse`, `psnr`, `ssim` | 构造一对一匹配代价矩阵的指标。 |
| `attack.recovery_match_objective` | `auto` | `auto`, `min`, `max` | 匹配时越小越好还是越大越好。 |
| `attack.recovery_success_metric` | `mse` | `mse`, `psnr`, `ssim` | 样本是否恢复成功的判定指标。 |
| `attack.recovery_success_objective` | `auto` | `auto`, `min`, `max` | 成功判定时越小越好还是越大越好。 |
| `attack.recovery_success_threshold` | `None` | 浮点数 | 显式恢复成功阈值；未设置时从默认阈值推导。 |
| `attack.success_mse_threshold` | `0.5` | 非负浮点数 | MSE 口径默认阈值。 |
| `attack.success_ssim_threshold` | 无 | 浮点数 | SSIM 口径默认阈值。 |
| `attack.success_rate_threshold` | `0.03` | `[0,1]` 浮点数 | 汇总层面的通过阈值。 |
| `attack.data_range` | `1.0` | 正浮点数 | PSNR / SSIM 计算时使用的数据范围。 |
| `attack.client_selection` | `all` | `all`, `first`, `round_robin` | 被攻击客户端选择策略。 |
| `attack.clients_per_round` | `1` | 正整数 | 每个攻击轮次攻击多少客户端。 |
| `attack.frequency_rounds` | `1` | 正整数 | 每隔多少轮触发一次攻击。 |
| `attack.sample_count` | `auto` | `auto` 或正整数 | 每客户端每轮发起多少次独立攻击。 |
| `attack.sample_count_cap` | `8` | 正整数 | 仅作为默认/兼容配置保留；当前主逻辑 `sample_count=auto` 默认解析为 1。 |
| `attack.max_samples` | `auto` | `auto` 或正整数 | 单次攻击联合重构多少样本。 |
| `attack.max_samples_cap` | `8` | 正整数 | `max_samples=auto` 时的上限。 |
| `attack.model_mode` | `train` | `train`, `eval` | 攻击时模型工作模式。 |
| `attack.local_optimizer` | `adam` | `adam`, `sgd` | `update_payload` 一步近似时假设的本地优化器。 |
| `attack.local_lr` | `0.001` | 正浮点数 | 一步近似本地更新学习率。 |
| `attack.local_optimizer_eps` | `1e-8` | 正浮点数 | `adam` 一步近似的 `eps`。 |
| `attack.async_enabled` | `false` | `true`, `false` | 是否异步执行攻击。 |
| `attack.async_workers` | `1` | 正整数 | 异步攻击 worker 数。 |
| `attack.async_max_pending_rounds` | `5` | 正整数 | 异步积压轮数上限。 |
| `attack.device` | `same` | `same`, `cpu`, `cuda:*` | 攻击设备。 |

### 3.10 `tracking`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `tracking.enabled` | `true` | `true`, `false` | 是否启用 wandb。 |
| `tracking.offline` | `true` | `true`, `false` | 是否使用 wandb offline。 |
| `tracking.project` | `federated-rare-earth` | 任意字符串 | wandb project 名。 |
| `tracking.name` | 无 | 任意字符串 | run 名。 |
| `tracking.group` | 无 | 任意字符串 | run group。 |
| `tracking.job_type` | 无 | 任意字符串 | job type。 |
| `tracking.tags` | 无 | 字符串列表 | wandb tags。 |

### 3.11 `grpc`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `grpc.address` | `0.0.0.0:50051` | `host:port` | 服务器监听地址。 |
| `grpc.server_address` | `127.0.0.1:50051` | `host:port` | 客户端连接地址。 |
| `grpc.poll_seconds` | `1.0` | 正浮点数 | 客户端轮询间隔。 |
| `grpc.max_message_mb` | `256.0` | 正浮点数 | gRPC 最大消息大小。 |

### 3.12 `artifacts`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `artifacts.config_formats` | `[yaml]` | `yaml`, `json`, `toml` 组成的列表 | 启动时保存哪些配置格式。 |
| `artifacts.save_every_rounds` | `0` | 非负整数 | 周期性快照间隔；`0` 表示关闭。 |

## 4. 方法专用配置块

### 4.1 `privacy`

当前主要由 `secure_quantized_fedavg` 使用。

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `privacy.clip_norm` | 无 | 非负浮点数 | 客户端上传前本地更新裁剪阈值。 |
| `privacy.noise_multiplier` | 无 | 非负浮点数 | 客户端上传前加噪强度。 |
| `privacy.delta` | 无 | 正浮点数 | 预留隐私参数，可由实验文档使用。 |

### 4.2 `adaptive_clipped_rdp`

当前由 `adaptive_clipped_rdp_fedavg` 使用。

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `noise_multiplier` | 无 | 非负浮点数 | 追加噪声强度。 |
| `reference_clip_norm` | 无 | 正浮点数 | 参考裁剪阈值。 |
| `min_clip_norm` | 无 | 正浮点数 | 裁剪阈值下界。 |
| `max_clip_norm` | 无 | 正浮点数 | 裁剪阈值上界。 |
| `clip_factor` | 无 | 正浮点数 | 裁剪阈值调整因子。 |
| `rdp_alpha` | 无 | 大于 1 的浮点数 | RDP accountant 使用的 alpha。 |
| `delta` | 无 | 正浮点数 | DP delta。 |
| `total_clients` | 无 | 正整数 | 总客户端数。 |
| `seed` | 无 | 整数 | 随机种子。 |

### 4.3 `ega`

当前由 `ega_fedavg` 使用，字段较多。

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `artifact_path` | 无 | 路径 | 预训练 EGA codec 路径。 |
| `num_clients` | 自动推断 | 正整数 | codec 初始化时的客户端数。 |
| `block_size` | 无 | 正整数 | 编码块大小。 |
| `encoded_dim` | 无 | 正整数 | 编码维度。 |
| `hidden_dim` | 无 | 正整数 | codec 隐层维度。 |
| `residual_blocks` | 无 | 非负整数 | 残差块数。 |
| `quantization_level` | 无 | 正整数 | 编码量化级数。 |
| `encode_buffers` | 无 | `true`, `false` | 是否编码 buffer 参数。 |
| `normalization` | 无 | 正浮点数 | 归一化尺度。 |
| `initial_normalization` | 无 | 正浮点数 | 初始归一化尺度。 |
| `min_normalization` | 无 | 正浮点数 | 归一化下界。 |
| `normalization_strategy` | 无 | 当前常用 `ema_reported_client_max_abs` | 归一化更新策略。 |
| `normalization_ema` | 无 | `[0,1]` 浮点数 | EMA 系数。 |
| `encoded_dtype` | 无 | 常用 `int8` | 编码后的整数类型。 |
| `encoded_stochastic_rounding` | 无 | `true`, `false` | 编码量化是否随机舍入。 |
| `encoded_noise_std` | 无 | 非负浮点数 | 编码噪声强度。 |
| `error_feedback` | 无 | `true`, `false` | 是否启用误差反馈。 |
| `pretrain.device` | 继承 runtime | 设备字符串 | 预训练设备。 |
| `pretrain.epochs` | 无 | 正整数 | codec 预训练轮数。 |
| `pretrain.patience` | 无 | 正整数 | codec 预训练早停耐心值。 |
| `pretrain.min_delta` | 无 | 非负浮点数 | codec 预训练最小改进阈值。 |
| `pretrain.batch_size` | 无 | 正整数 | codec 预训练 batch size。 |
| `pretrain.lr` | 无 | 正浮点数 | codec 预训练学习率。 |
| `pretrain.train_groups` | 无 | 正整数 | 预训练训练样本组数。 |
| `pretrain.val_groups` | 无 | 正整数 | 预训练验证样本组数。 |
| `pretrain.seed` | 无 | 整数 | codec 预训练随机种子。 |

说明：EGA 对上传侧做编码压缩；下载侧使用框架公共的 `model` 下发逻辑，因此没有独立的 `ega.download_*` 配置项。

## 5. 数据与模型块的任务专用字段

### 5.1 forecasting 任务常见 `data` / `model`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `data.split_dir` | 无 | 路径 | 稀土等已划分数据集目录。 |
| `data.csv_path` | 无 | 路径 | 原始 CSV 路径；仅部分准备流程使用。 |
| `data.clients` | 无 | 客户端 ID 列表 | 联邦客户端名称。 |
| `data.seq_len` | `21` | 正整数 | 输入窗口长度。 |
| `data.pred_len` | `7` | 正整数 | 预测窗口长度。 |
| `data.batch_size` | 无 | 正整数 | DataLoader batch size。 |
| `data.shuffle_train` | 无 | `true`, `false` | 训练集是否打乱。 |
| `data.num_workers` | 无 | 非负整数 | DataLoader worker 数。 |
| `model.name` | `mlp` | `mlp`, `lstm`, `patchtst` | forecasting 模型名。 |
| `model.channels` | `1` | 正整数 | 输入通道数。 |
| `model.hidden_size` | `64` | 正整数 | `mlp` / `lstm` 隐层大小。 |
| `model.num_layers` | `1` | 正整数 | `lstm` 层数。 |
| `model.dropout` | `0.1` | `[0,1)` 浮点数 | dropout。 |
| `model.patch_len` | `16` | 正整数 | `patchtst` patch 长度。 |
| `model.stride` | `8` | 正整数 | `patchtst` patch 步长。 |
| `model.d_model` | `384` | 正整数 | `patchtst` 隐层维度。 |
| `model.n_heads` | `4` | 正整数 | `patchtst` 注意力头数。 |
| `model.e_layers` | `2` | 正整数 | `patchtst` encoder 层数。 |
| `model.d_ff` | `2048` | 正整数 | `patchtst` FFN 维度。 |
| `model.factor` | `3` | 正整数 | `patchtst` 配置参数。 |
| `model.activation` | `gelu` | 如 `gelu` | `patchtst` 激活函数。 |

### 5.2 classification 任务常见 `data` / `model`

| 字段 | 默认值 | 可选值 | 作用 |
| --- | --- | --- | --- |
| `data.dataset_name` | 无 | `mnist`, `cifar10` 等 | 图像数据集名称，用于部分默认推断。 |
| `data.split_dir` | 无 | 路径 | 已划分图像数据目录。 |
| `data.clients` | 无 | 客户端 ID 列表 | 联邦客户端名称。 |
| `data.image_shape` | 按数据集推断 | 如 `[1,28,28]`, `[3,32,32]` | 图像形状。 |
| `data.num_classes` | `10` | 正整数 | 类别数。 |
| `data.batch_size` | 无 | 正整数 | batch size。 |
| `data.shuffle_train` | 无 | `true`, `false` | 训练是否打乱。 |
| `data.num_workers` | 无 | 非负整数 | DataLoader worker 数。 |
| `model.name` | `small_cnn` | `small_cnn`, `cnn`, `convnet`, `mlp`, `flatten` | 分类模型名。 |
| `model.hidden_channels` | `32` | 正整数 | `small_cnn` 通道宽度。 |
| `model.hidden_size` | `128` | 正整数 | `mlp` 隐层大小。 |
| `model.dropout` | `0.1` | `[0,1)` 浮点数 | dropout。 |
| `model.image_shape` | 无 | 三元列表 | 可覆盖 `data.image_shape`。 |
| `model.num_classes` | `10` | 正整数 | 可覆盖分类数。 |

## 6. 当前规范配置目录

为了便于维护，当前建议优先使用下面这些 canonical 配置路径：

- `configs/rare/*.yaml`
- `configs/mnist/*.yaml`
- `configs/cifar10/*.yaml`

当前顶层旧路径仍然保留为兼容 wrapper，例如：

- `configs/fedavg.yaml -> configs/rare/fedavg.yaml`
- `configs/mnist_classification.yaml -> configs/mnist/fedavg.yaml`
- `configs/cifar10_ega_classification.yaml -> configs/cifar10/ega.yaml`

## 7. 示例

### 7.1 稀土 FedAvg

```yaml
includes:
  - rare/fedavg.yaml
```

### 7.2 MNIST Top-k

```yaml
includes:
  - mnist/topk.yaml
```

### 7.3 CIFAR10 EGA

```yaml
includes:
  - cifar10/ega.yaml
```
