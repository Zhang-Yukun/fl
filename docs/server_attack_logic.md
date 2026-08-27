# Server-Side Attack Logic

本文档梳理当前框架中的服务器端重构攻击逻辑，内容以当前代码为准。

## 1. 攻击方与攻击时机

当前攻击方是服务器端，假设为 honest-but-curious server。

每轮联邦训练中的关键顺序是：

1. 服务器保存聚合前全局模型 `round_base_state`
2. 客户端基于该模型做本地训练
3. 客户端上传本轮更新 payload
4. 服务器完成聚合
5. 服务器做验证评测
6. 服务器对选中的客户端执行同步或异步攻击

因此攻击发生在每轮客户端上传之后，不参与聚合，也不改变本轮验证指标。

## 2. 当前攻击目标

当前框架只支持 `attack.target_type=update_payload`。

服务器攻击看到的是方法对象导出的 `attack view`，也就是协议中服务器真实可见的上传内容：

- dense 算法：dense update
- sparse 算法：解压后的稠密 update 视图
- quantized 算法：反量化后的 update 视图
- encoded 算法：协议真正暴露给服务器的可解码视图

也就是说，当前实现直接攻击服务器实际截获到的上传内容，而不是额外采样梯度。

## 3. 真实参考样本与恢复集合

对于被攻击客户端，攻击任务会保存：

- `real_x`、`real_y`：本次攻击选中的真实样本批次
- `reference_inputs`、`reference_targets`：该客户端训练集参考集合

`sample_count` 控制一轮里对同一客户端发起多少次独立攻击，默认 `auto=1`。

`max_samples` 控制单次攻击联合重构多少个样本；当配置为 `auto` 时，会取该客户端可用训练样本数，并再受 `max_samples_cap` 限制。

因此当前评估语义是：服务器基于一次上传 payload，尝试恢复一个给定预算大小的样本集合，再与客户端训练参考集合做匹配。

## 4. DLG 与 iDLG 的当前含义

每个攻击任务都会按配置顺序执行 `DLG` 与 `iDLG`。

- `DLG`：同时优化 `dummy_x` 和 `dummy_y`
- `iDLG`：
  - 分类任务下，按照原始 iDLG 论文的最后一层梯度符号逻辑推断伪标签，只优化 `dummy_x`
  - 时序 / 回归任务下，退化为与 DLG 相同的目标优化形式

当前优化目标固定为：比较 dummy batch 诱导出的一步近似本地更新，与截获 update payload 的距离。

近似本地更新不是完整重跑一个 epoch，而是一步近似：

- `SGD`: `-lr * grad`
- `Adam`: `-lr * grad / (|grad| + eps)`

攻击优化器当前只支持 `adam`。

## 5. 当前攻击指标

单次攻击结果至少会记录：

- `nearest_client_train_mse`
- `matched_reference_indices`
- `matched_reference_metric_name`
- `matched_reference_metric_value`
- `recovered_count`
- `reconstructed_count`
- `reference_count`
- `budget_recovered_fraction`
- `coverage_recovered_fraction`
- `psnr`
- `ssim`
- `objective_mse`
- `iterations`
- `time_seconds`

其中集合恢复统计的含义是：

- `budget_recovered_fraction = recovered_count / reconstructed_count`
- `coverage_recovered_fraction = recovered_count / reference_count`

当前正式实验的默认主指标是 `budget_recovered_fraction`。其隐私解释方向是 `lower_is_more_private`，也就是恢复率越低越安全。

## 6. 成功判定

攻击结果会先对重构样本与参考集合做一对一匹配：

- 用 `attack.recovery_match_metric` 构造匹配代价矩阵
- 用 `attack.recovery_match_objective` 指定匹配时取大还是取小更优
- 用 `attack.recovery_success_metric` 和阈值判断匹配后的样本是否算恢复成功

如果显式配置了 `attack.recovery_success_threshold`，则使用该阈值；否则按所选成功指标的默认阈值推导。

单条攻击记录的 `success` 表示这条记录中的所有重构样本都达到恢复阈值；汇总里的 `attack_success_rate` 是按记录统计的成功率。

## 7. 同步、异步与独立回放

同步攻击：当前轮训练结束后立即执行，主流程等待攻击完成。

异步攻击：当前轮只保存攻击任务快照，后台线程池执行攻击，训练结束前统一收尾。

另外，服务器还会按攻击频率把截获到的更新保存到输出目录下的 `saved_updates/<client_id>/round_xxxx.pt`。`replay_saved_update_attacks.py` 会扫描这些保存结果并按当前攻击配置顺序重放全部已启用攻击，`replay_saved_update_dlg.py` 与 `replay_saved_update_idlg.py` 则分别只重放单一攻击，产物格式与在线攻击保持一致。

## 8. 一句话总结

当前框架中的服务器端攻击逻辑是：服务器在每轮收到客户端上传后，针对真实可见的 `update_payload` 运行 DLG / iDLG，并以集合恢复率和匹配质量作为主要隐私泄露评估指标。
