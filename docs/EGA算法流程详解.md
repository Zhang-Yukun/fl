# EGA 算法流程详解

本文档面向当前仓库中的 `ega_fedavg` 实现，尽量完整地说明它在本项目里的真实执行逻辑，而不是只给论文层面的抽象描述。

相关代码入口：
- [fedlab/federated/methods/encoded.py](../fedlab/federated/methods/encoded.py)
- [fedlab/modeling/ega.py](../fedlab/modeling/ega.py)
- [configs/ega.yaml](../configs/ega.yaml)

配套图：
- [EGA算法交互图.drawio](./EGA算法交互图.drawio)

## 1. EGA 在这个项目里到底是什么

在这个仓库里，`ega_fedavg` 不是“所有参数都统一丢进一个编码器”这么简单。它的真实结构是：

1. 只对 **trainable 参数更新** 走 EGA 编码上传支路。
2. **buffer / 非 trainable 状态更新** 不进 EGA codec，而是单独作为 dense `aggregation_state` 携带。
3. 服务器端先对所有客户端的 `ega_payload` 解码并取平均，恢复 trainable 部分的平均更新。
4. 然后把 buffer 的 dense update 再按客户端权重加回去，组成完整的模型更新。
5. 最终仍然是一个 FedAvg 风格的“加到 protocol base 上”的更新过程。

所以 EGA 在本项目里的本质是：

- trainable update：编码压缩
- untrainable / buffer update：单独 dense 旁路
- server aggregation：先 decode mean，再 merge dense buffers

## 2. 本文采用的参考配置与流程前提

下面这份配置画像就是本文和配套图唯一采用的 EGA profile。后文所有流程描述都按这组参数展开，不再单独讨论其他配置分支：

```yaml
ega:
  artifact_path: artifacts/ega/ega_ed128_dm_ega_pc_q127.pt
  block_size: 256
  encoded_dim: 128
  hidden_dim: 1024
  residual_blocks: 2
  quantization_level: 127
  encode_buffers: false
  normalization: 0.00025
  initial_normalization: 0.00025
  min_normalization: 1.0e-06
  normalization_strategy: ema_reported_client_max_abs
  normalization_ema: 0.9
  encoded_dtype: int8
  encoded_stochastic_rounding: false
  encoded_noise_std: 0.0
  download_dtype: float32
  download_method: dense
  download_predictive_coding: false
  download_stochastic_rounding: false
  download_trainable_only: false
  download_encoded_dtype: int8
  error_feedback: true
  pretrain:
    epochs: 100
    patience: 20
    min_delta: 0.0
    batch_size: 128
    lr: 0.0005
    train_groups: 30000
    val_groups: 15000
    seed: 2026
```

与框架层联动的运行语义是：

- `federated.algorithm = ega_fedavg`
- `transport.upload_mode = update`
- `transport.download_mode = model`
- `evaluation.mode = protocol`

结合上面这组参数，本文中的整套流程固定为：

1. 服务端下发完整模型语义，而不是增量下载。
2. 客户端上传的是 update 语义，而不是完整模型。
3. 上传侧 EGA 编码使用 `block_size=256`、`encoded_dim=128`、`quantization_level=127`。
4. trainable update 的传输侧 encoded dtype 为 `int8`。
5. 下载侧不走 EGA 编码链路，而是 `download_method=dense`、`download_dtype=float32`。
6. error feedback 开启。
7. normalization 采用 `ema_reported_client_max_abs`，衰减系数 `0.9`。
8. 评测看 protocol 模型，不展开 oracle 分支。

## 3. 关键符号

下面这些符号最有用：

- `G_r`：第 `r` 轮开始时服务器持有的 protocol 全局模型。
- `C_{i,r}`：客户端 `i` 本轮真正收到并用于训练的模型。
- `L_{i,r}`：客户端 `i` 本地训练结束后的模型。
- `U_{i,r}`：客户端 `i` 的本地更新，默认可理解为 `L_{i,r} - C_{i,r}`。
- `P^{down}_{i,r}`：服务器发给客户端 `i` 的下载 payload。
- `P^{up}_{i,r}`：客户端 `i` 发给服务器的上传 payload。
- `ega_payload`：客户端上传的 EGA 编码负载，专门对应 trainable update。
- `aggregation_state`：客户端 alongside 上传的 dense buffer update。
- `U_hat_{i,r}`：服务器解码后能看到的协议可见 update。

在代码里，大体对应：

- `server.global_state` -> `G_r`
- `received_global_state` -> `C_{i,r}`
- `local_state` -> `L_{i,r}`
- `result.ega_payload` -> EGA 编码后的 trainable update payload
- `result.aggregation_state` -> dense buffer update

## 4. EGA codec 是什么

EGA codec 在 [fedlab/modeling/ega.py](../fedlab/modeling/ega.py) 里实现为 `EgaAutoEncoder`。

它是一个对称的 residual MLP autoencoder：

- 输入维度：`block_size`
- 编码维度：`encoded_dim`
- 隐层宽度：`hidden_dim`
- 中间残差块数：`residual_blocks`

更具体地说：

1. 先把一个扁平 update 向量按 `block_size` 切成多个 block。
2. 对每个 block 先做整数域随机量化。
3. 再把每个量化 block 输入到 `codec.encode_blocks(...)`，得到 encoded block。
4. 服务端聚合时对 encoded block 先求 mean，再用 `codec.decode_blocks(...)` 解码。

注意：这里并不是把多个客户端的原始 block 先平均再编码，而是：

- 每个客户端先独立 encode
- 服务器对 encoded representation 求平均
- 再 decode 平均后的 encoded representation

这正是当前 `decode_mean_encoded_payload(...)` 的语义。

## 5. codec 从哪里来

### 5.1 artifact 路径

本文配置把 artifact 路径固定为 `artifacts/ega/ega_ed128_dm_ega_pc_q127.pt`。

在实际运行中，这个相对路径会由 `resolve_ega_artifact_path(...)` 解析；如果配置里同时给了 `experiment.output_dir`，最终文件会落在对应输出目录下的相对位置。

### 5.2 载入逻辑

服务端在 `EGAFedAvgMethod.configure_server(...)` 里调用 `load_ega_codec(...)`。

`load_ega_codec(...)` 的逻辑是：

1. 检查 artifact 是否存在。
2. 不存在且 `allow_pretrain=True` 时，先做 synthetic pretraining。
3. 存在但结构 spec 和当前 config 不匹配时，重新训练。
4. 最终加载 checkpoint 里的 `state_dict`，构造 codec。

### 5.3 synthetic pretraining

预训练不是在真实模型 update 上做的，而是在 **随机整数域 block group** 上做的。

训练目标是：

- 输入：`[batch, num_clients, block_size]` 的随机整数张量
- target：各 client block 的均值
- model：`codec(batch)`
- loss：`MSELoss`

也就是说，这个 codec 被训练成：

- 接收多个客户端的 quantized block
- 通过 encode + mean + decode
- 逼近这些 block 的真实均值

这和服务端最终要做的“encoded mean aggregation”是一致的。

## 6. 服务端初始化阶段

`EGAFedAvgMethod.configure_server(...)` 会给 server 挂很多 EGA 状态：

- `server.ega_total_clients`
- `server.ega_codec`
- `server.ega_codec_bootstrap_payload`
- `server.ega_codec_bootstrap_pending`
- `server.ega_trainable_keys`
- `server.ega_normalization`
- `server.ega_received_global_states`

其中比较关键的是：

### 6.1 `ega_codec_bootstrap_payload`

服务端会把 codec 的 `state_dict` 和结构 spec 打包成 bootstrap payload。

这个 payload 会在首轮通过 `round_context` 发给客户端，用于客户端本地构造同一个 codec。

### 6.2 `ega_codec_bootstrap_pending`

这是一个只发一次的标志：

- 首轮前为 `True`
- 首轮聚合后在 `aggregate(...)` 里设为 `False`
- 之后的 `round_context` 就不再带 codec payload

测试也验证了这一点。

### 6.3 `ega_normalization`

这是每轮用于 stochastic quantization 的 normalization 标量。

本文配置里：

- `ega.initial_normalization = 0.00025`
- `ega.normalization = 0.00025`
- `ega.min_normalization = 1.0e-06`

因此服务端起始 normalization 就是 `0.00025`，之后再在每轮聚合后更新。

## 7. 客户端初始化阶段

`EGAFedAvgMethod.configure_client(...)` 给客户端挂了这些状态：

- `client.ega_codec = None`
- `client.ega_codec_ready = False`
- `client.ega_trainable_keys`
- `client.ega_residual = None`

其中：

### 7.1 `ega_trainable_keys`

客户端会先建一个模板模型，然后用 `serialize_trainable_model(template)` 记录哪些键属于 trainable state。

之后在上传阶段就靠这个列表把：

- trainable update
- buffer / untrainable update

拆成两条支路。

### 7.2 `ega_residual`

这是 error feedback 用的残差缓存。

由于本文配置中 `ega.error_feedback = true`，客户端会把上轮残差加回当前 trainable update，再做编码；编码之后再更新新的 residual。

## 8. `round_context` 里到底有什么

服务端每轮通过 `build_round_context(...)` 广播：

- `ega_normalization`
- 首轮才有的 `ega_codec_payload`

因此 EGA 的 download side 不只是普通的 global model，还依赖一个额外的 context。

这也是为什么：

- 下载参数字节不能只看 `download_state`
- 还要把 `round_context` 的辅助字节算进去

测试 `test_single_node_ega_transport_counts_round_context_bytes` 明确验证了这一点。

## 9. 下载路径：客户端如何得到 `C_{i,r}`

EGA 下载逻辑都集中在 `_prepare_received_global_state(...)`。

这个函数的目标是同时返回两样东西：

1. 实际下载 payload `download_state`
2. 客户端最终拿去训练的 `received_state`

### 9.1 本文采用的下载路径

在本文采用的配置下：

- `transport.download_mode = model`
- `ega.download_method = dense`
- `ega.download_dtype = float32`
- `ega.download_predictive_coding = false`
- `ega.download_trainable_only = false`
- `ega.download_encoded_dtype = int8`

因此下载语义可以直接理解为：

1. 服务器目标下发模型就是当前 `global_state`。
2. 因为 `download_mode=model`，payload 语义本身就是完整模型。
3. `download_method=dense`，因此不走 EGA 编码下载。
4. `quantize_state_update(..., dtype=download_dtype)` 在下载链路上做数值类型处理。
5. 客户端再通过 `dequantize_state_update(...)` 恢复出 `received_state`。

由于 `download_dtype=float32`，这条下载路径在数值上接近无损。客户端收到的重点不是压缩下载，而是：

- 收到完整 model 语义的 `download_state`
- 结合 `round_context` 中的 `ega_normalization`
- 在本地恢复出用于训练的 `received_state`

## 10. 客户端本地训练后如何构造上传内容

上传主逻辑在 `EGAFedAvgMethod.client_update(...)`。

### 10.1 先拆 trainable 和 untrainable

代码先拿到：

- `trainable_state = serialize_trainable_model(model)`
- `untrainable_state = serialize_untrainable_model(model)`

再基于 `received_global_state` 构造：

- `global_trainable_state`
- `global_untrainable_state`

然后分别相减得到：

- `trainable_update`
- `buffer_update`

### 10.2 buffer update 会先做一次裁剪式清理

`buffer_update` 会经过 `_drop_zero_state(...)`。

作用是：

- 去掉绝对值最大值不超过 `buffer_tolerance` 的张量
- 避免把几乎全零的 buffer 改动也带着上传

### 10.3 error feedback

本文配置中 `ega.error_feedback = true`，因此客户端在存在上轮 residual 时执行：

- `effective_update = trainable_update + ega_residual`

首轮没有 residual 时，就有：

- `effective_update = trainable_update`

这一步发生在真正编码前。

### 10.4 contribution scale

上传前还会计算：

`contribution_scale = client_samples / total_train_samples * total_clients`

直观上这是把“按样本数的 FedAvg 权重”折进每个客户端上传的数值尺度里，使服务端直接对 encoded payload 求平均时，能够恢复出加权聚合语义。

### 10.5 normalization 从哪里来

在本文流程里，客户端每轮都从 `round_context['ega_normalization']` 里拿 normalization。

在起始轮次，这个量与配置中的 `ega.initial_normalization = 0.00025` 一致；后续轮次则使用服务端更新后的值。最终还会与 `ega.min_normalization = 1.0e-06` 比较取 max，避免除零或极小归一化尺度。

## 11. `encode_state_update(...)` 内部到底做了什么

这是 EGA 的核心函数。

给定一个 dense state update，它会：

1. `_flatten_state(update)`
   - 保留参数名和 shape
   - 把所有 tensor flatten 成一个长向量

2. `scaled = flat * contribution_scale`
   - 先把客户端加权系数折进去

3. `pack_flat_blocks(scaled, block_size)`
   - 把长向量补零并切成固定大小的 block

4. `stochastic_quantize_block_vector(...)`
   - 依据 `quantization_level` 和 `normalization`
   - 把每个 block 的实数值映射到 `[-s, s]` 整数域
   - 使用随机舍入

5. `codec.encode_blocks(...)`
   - 把每个 quantized block 送进 EGA encoder
   - 得到 `encoded_blocks`

6. 本文配置里 `encoded_noise_std = 0.0`
   - 因此这一步不额外加噪声

7. `quantize_encoded_blocks(...)`
   - 对 encoded representation 本身再做一次传输侧量化
   - 本文配置里使用 `int8`

8. 组装成 `EncodedStatePayload`
   - 包括 names, shapes, encoded blocks, normalization, contribution_scale, encoded dtype, observed absmax 等元信息

也就是说，这里其实有两层“量化”：

- 第一层：block value 量化到整数域，进入 codec
- 第二层：encoded representation 自身再做传输量化

## 12. error feedback 是怎么更新 residual 的

由于本文配置开启了 error feedback：

1. 客户端先用刚生成的 `payload` 调 `decode_mean_encoded_payload([payload], codec)`。
2. 因为这里只有一个 payload，所以这是“单客户端近似重建”。
3. 再除回 `contribution_scale`，得到 `approx_raw`。
4. 最后：
   - `ega_residual = effective_update - approx_raw`

这样，下轮编码前 residual 会被加回去，尽量补偿上一轮编码误差。

## 13. 服务端如何聚合 EGA 上传

服务端的 `aggregate(...)` 逻辑分三段。

### 13.1 先对所有 `ega_payload` 解码平均

服务端收集所有 `payloads = [result.ega_payload for result in results]`，然后：

- `averaged_update = decode_mean_encoded_payload(payloads, server.ega_codec)`

这一步恢复的是：

- trainable 部分的平均 update

### 13.2 再把 buffer update 按权重加回

如果某个客户端 `result.aggregation_state` 不为空：

- 服务器按 `weights = num_samples / total_samples`
- 把每个 dense buffer update 加回 `averaged_update`

于是就得到了 full update。

### 13.3 聚合不是直接加到 `round_base_state`

服务端最终不是机械地做：

- `server.global_state = round_base_state + full_update`

而是先求一个 **protocol base state**：

- `weighted_protocol_base_state(server, results, round_base_state, round_index, {})`

这里会考虑“每个客户端本轮实际看到的模型”是否一致。

这点非常关键，因为 EGA 的下载链路可能是有损的、也可能是预测式的，所以：

- client-visible base 不一定严格等于 server raw global state

最终更新是：

- `server.global_state = protocol_base_state + full_update`

## 14. normalization 是怎么更新的

聚合后，服务端按照 `ega.normalization_strategy = ema_reported_client_max_abs` 更新 `server.ega_normalization`。

具体做法是：

1. 先收集各客户端 payload 中的 `payload.observed_update_absmax`。
2. 取其中最大者，记为 `observed`。
3. 再按 `normalization_ema = 0.9` 做指数滑动更新：
   - `new_norm = 0.9 * previous + 0.1 * observed`

这里的 `observed_update_absmax` 是客户端在 `encode_state_update(...)` 中记录的 `scaled = flat * contribution_scale` 的绝对值最大值。

## 15. 服务端攻击视图是什么

EGA 的攻击视图不是直接拿某个客户端的 raw update，而是通过：

- `decode_attack_view_from_mean_difference(payloads, target_index, server.ega_codec)`

恢复。

它的思路是：

1. 先求所有客户端 encoded blocks 的 mean。
2. 再把目标客户端替换成 zero block encoded representation。
3. 再求“去掉目标客户端后的 mean”。
4. decode 两者的差值。

这得到的是一个“honest-but-curious server 可见的目标客户端贡献视图”。

然后代码还会：

- 把目标客户端的 `aggregation_state`（也就是 dense buffer update）合并进去。

所以最终 EGA attack view =

- 通过 mean difference 恢复出的 trainable contribution
- 加上该客户端的 buffer dense update

## 16. 通信量在 EGA 里怎么算

### 16.1 上传参数字节

客户端上传阶段，`ClientResult` 里会设置：

- `upload_bytes = payload.nbytes + buffer_bytes`
- `upload_parameters = payload.num_parameters + buffer_parameters`

其中：

- `payload.nbytes` 来自 `EncodedStatePayload.nbytes`
- `buffer_bytes` 来自 dense buffer update 的 `state_num_bytes(...)`

### 16.2 下载参数字节

EGA 下载阶段还要额外把 `round_context` 算进参数字节：

- `parameter_download_bytes = state_num_bytes(download_state) + auxiliary_payload_num_bytes(round_context)`

因此首轮通常会更大，因为首轮 `round_context` 里包含完整的 `ega_codec_payload`。

### 16.3 单节点 transport bytes

单节点没有真实网络，但框架仍会用序列化估算 transport envelope：

- `estimate_download_transport_bytes(...)`
- `estimate_upload_transport_bytes(...)`

### 16.4 gRPC transport bytes

多节点路径下，真实传输字节来自：

- `FederatedRpcClient.get_global()`
- `FederatedRpcClient.submit_update()`

周围的 `sent_bytes` / `received_bytes` 计数。

## 17. EGA 与 protocol 评测的关系

本文配置使用 `evaluation.mode=protocol`，所以主要看的是：

- `server.global_state`

因此本文 EGA 流程的评测重点放在：

- protocol 聚合语义
- protocol 测试指标
- encoded upload 的通信压缩效果

## 18. 为什么本文的下载路径不是 encoded download

本文采用的配置中：

- `ega.download_method = dense`
- `download_dtype = float32`
- `download_predictive_coding = false`
- `download_trainable_only = false`

这意味着下载侧使用 dense model path。

这背后的工程含义是：

1. 主目标是研究 **上传侧** 编码聚合。
2. 下载侧先保持稳定、简单。
3. 因此图和正文都不展开下载编码分支。

## 19. EGA 在当前框架里的完整一轮流程总结

按本文这组配置，一轮 EGA 可以概括为：

1. 服务端加载或预训练 codec，并维护 `ega_normalization`。
2. 首轮通过 `round_context` 把 codec bootstrap payload 发给客户端。
3. 服务端按 model 语义下发全局模型 `G_r`。
4. 客户端重建 `C_{i,r}`，本地训练得到 `L_{i,r}`。
5. 客户端把 trainable update 走 EGA 编码支路，把 buffer update 走 dense 支路。
6. 客户端上传 `ega_payload + buffer_update`。
7. 服务端先 decode mean encoded payload，恢复 trainable 平均 update。
8. 服务端再把 dense buffer update 按权重补回。
9. 服务端基于 client-visible protocol base 更新 `server.global_state`。
10. 服务端更新 `ega_normalization`。
11. 服务端按 protocol 口径做评测。
12. 服务端在 round 尾部基于 EGA attack view 触发攻击评估。
13. 服务端记录通信量，其中首轮还包含 codec bootstrap 的 round_context 开销。

## 20. 最容易误解的几个点

### 20.1 EGA 不是“整个模型都编码上传”

不是。当前实现只对 trainable update 做编码；buffer update 走 dense 旁路。

### 20.2 EGA 这里的下载不是 encoded download

不是。本文采用的配置明确使用 `download_method=dense`。

### 20.3 服务端不是 decode 每个客户端再求平均

不是。它是先对各客户端 encoded representation 求 mean，再 decode。

### 20.4 error feedback 不是在服务端做的

不是。residual 完全保存在客户端，并在下轮编码前加回。

### 20.5 首轮 download 通常更大

是的，因为首轮 `round_context` 里要附带 codec bootstrap payload。

## 21. 推荐阅读顺序

如果你要继续深入，我建议按这个顺序看代码：

1. [configs/ega.yaml](../configs/ega.yaml)
2. [fedlab/federated/methods/encoded.py](../fedlab/federated/methods/encoded.py)
3. [fedlab/modeling/ega.py](../fedlab/modeling/ega.py)
4. [tests/federated/test_algorithms.py](../tests/federated/test_algorithms.py) 中所有 `ega_` 相关测试
5. [tests/modeling/test_ega.py](../tests/modeling/test_ega.py)

