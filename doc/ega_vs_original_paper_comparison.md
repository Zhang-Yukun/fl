# 当前仓库 EGA 与原始 EGA 论文对比表

本文档对比当前仓库中的 `ega_fedavg` 实现，与原始论文 *Encoded Gradients Aggregation against Gradient Leakage in Federated Learning* 的主要差异，重点说明本项目在工程实现和通信设计上的扩展点。

相关代码入口：

- [fedlab/federated/methods/encoded.py](../fedlab/federated/methods/encoded.py)
- [fedlab/modeling/ega.py](../fedlab/modeling/ega.py)
- [configs/common/ega/default.yaml](../configs/common/ega/default.yaml)

## 对比总表

| 对比维度 | 原始论文 EGA | 当前仓库 `ega_fedavg` | 当前仓库的改进点 |
| --- | --- | --- | --- |
| 压缩对象 | 重点讨论梯度经过编码后的聚合与隐私防护。 | 实际上传的是 **trainable 参数更新量**，而不是直接暴露原始梯度对象。 | 把方法落到了更稳定的 FedAvg 更新语义上，更容易直接接入现有联邦训练流程。 |
| 聚合语义 | 关注编码表示上的聚合与恢复。 | 客户端先对 trainable update 编码，服务端对 `ega_payload` 做均值解码，再恢复平均更新。 | 明确与现有 FedAvg 聚合对齐，服务端最终得到的是可直接加到全局模型上的平均更新。 |
| 非训练参数处理 | 论文重点不在 buffer / 非训练参数的工程分流。 | `encode_buffers: false` 时，buffer / 非 trainable 状态不进 codec，而是通过 `aggregation_state` 单独上传。 | 将 trainable 与非 trainable 状态拆路处理，避免把 BN 统计量、running buffer 等一并塞进编码器导致失真或冗余传输。 |
| 编码后传输格式 | 更强调“编码表示”本身的保护与聚合。 | 编码结果还能继续做低比特传输，默认 `encoded_dtype: int8`。 | 在编码之后再做一次传输侧量化，进一步压缩上传字节数，这是对论文实现层面的明显增强。 |
| 上下行通信设计 | 更偏上传侧编码聚合问题。 | 当前实现显式建模了上传 `ega_payload`、首轮 `ega_codec_payload`、每轮 `ega_normalization` 等真实通信对象。 | 不只是讨论算法抽象，还把方法扩展成完整的 gRPC 训练协议，便于统计 parameter、transport 和外部抓包三种通信量。 |
| 误差补偿 | 原始思想核心在编码聚合本身。 | 默认开启 `error_feedback: true`，客户端保留 `ega_residual`，下一轮把残差加回更新再编码。 | 通过误差反馈降低反复编码带来的累积失真，增强多轮训练稳定性。 |
| 归一化稳定化 | 论文层面通常不会细化成完整工程策略集合。 | 默认启用 `normalization`、`initial_normalization`、`min_normalization` 和 `normalization_strategy: ema_reported_client_max_abs`。 | 增加了可持续更新的归一化控制，避免编码输入尺度在不同轮次剧烈波动。 |
| codec 启动方式 | 更偏方法描述。 | 服务端会加载或预训练 codec，并在首轮通过 `round_context["ega_codec_payload"]` 下发给客户端。 | 补齐了真实系统中的 codec 启动、分发与首轮 bootstrap 机制。 |
| 预训练方式 | 论文重点在方法有效性，不一定规定当前仓库这种预训练工程。 | 当前仓库会对 codec 做 synthetic pretraining，使其学习“编码后求均值再解码”这一目标。 | 将理论方法固定成可重复复用的 artifact，降低每次实验重新构造 codec 的不确定性。 |
| 攻击视图支持 | 论文直接围绕梯度泄露问题展开。 | 当前仓库额外提供 `decode_attack_view_from_mean_difference(...)` 等接口，方便在框架内分析攻击可恢复视图。 | 把算法实现与攻击分析工具链打通，便于直接评估压缩与泄露风险之间的关系。 |
| 默认配置倾向 | 论文更强调方法思想。 | 当前默认值为 `encode_buffers: false`、`encoded_dtype: int8`、`error_feedback: true`、`encoded_noise_std: 0.0`。 | 当前实现默认优先追求“稳定通信压缩 + 可运行性”，而不是只保留最原始的论文形态。 |

## 可直接概括为四点的改进

| 改进点 | 一句话说明 | 当前默认是否启用 |
| --- | --- | --- |
| 可训练参数编码压缩 | 只对 trainable 参数更新走 EGA 编码主支路。 | 是 |
| 非训练参数按需同步 | buffer / 非 trainable 状态默认不编码，只在有变化时走旁路上传。 | 是 |
| 编码后低比特传输 | 编码结果默认再量化成 `int8` 传输，进一步减少字节数。 | 是 |
| 稳定化机制 | 通过误差反馈和 EMA 归一化控制，减轻多轮编码误差累积。 | 是 |

## 当前默认配置下，这四点分别对应什么

| 类别 | 关键参数 | 当前默认值 | 含义 |
| --- | --- | --- | --- |
| 编码主支路 | `ega.block_size` / `ega.encoded_dim` / `ega.hidden_dim` | `256` / `144` / `2048` | 用于定义 EGA codec 的输入分块与编码维度。 |
| 非训练参数分流 | `ega.encode_buffers` / `ega.buffer_tolerance` | `false` / `0.0` | 默认 buffer 不进入编码器，只有非零变化的 buffer update 才单独上传。 |
| 传输侧量化 | `ega.encoded_dtype` / `ega.quantization_level` | `int8` / `159` | 编码结果默认按低比特形式传输，而不是直接传 float32 编码向量。 |
| 稳定化 | `ega.error_feedback` / `ega.normalization_strategy` / `ega.normalization_ema` | `true` / `ema_reported_client_max_abs` / `0.98` | 通过残差补偿和 EMA 归一化，提升多轮训练时的稳定性。 |

## 一段适合直接引用的总结

与原始 EGA 论文相比，当前仓库中的 `ega_fedavg` 已不只是“把梯度编码后再聚合”的论文级方法描述，而是扩展成了一套可直接运行在联邦训练框架中的工程实现。它在保留编码聚合核心思想的基础上，进一步加入了四个更贴近真实训练系统的设计：第一，只对可训练参数更新做编码压缩；第二，将非训练参数从编码主支路中拆分出来并按需同步；第三，对编码结果继续做 `int8` 级别的低比特传输压缩；第四，引入误差反馈与 EMA 归一化等稳定化机制。也因此，当前实现相比原始论文更强调通信效率、训练稳定性以及与完整系统协议的对齐。 
