# Server-Side Attack Logic

本文档梳理当前框架中的服务器端重构攻击逻辑，便于对照实现、配置与实验结果。

## 1. 攻击方是谁

当前实现中的攻击方是服务器端，假设为 honest-but-curious server。

服务器在每轮联邦训练中能够看到客户端上传的单独更新内容，因此可以基于这些上传内容发起重构攻击。

## 2. 攻击在什么时候发生

当前单节点与多节点实现的攻击时机一致，整体顺序如下：

1. 服务器保存本轮聚合前的全局模型 `round_base_state`
2. 客户端基于该模型进行本地训练
3. 客户端上传本轮更新 payload
4. 服务器完成聚合
5. 服务器在验证集上评估聚合后的全局模型
6. 服务器构造攻击任务并执行攻击

因此，当前攻击不是发生在最终测试阶段，而是发生在每轮客户端上传之后。

## 3. 攻击目标是什么

当前框架把每轮联邦语义显式拆成三层：

- `aggregation payload`：真正参与服务器聚合的上传内容
- `attack view`：服务器攻击端真实可见的内容
- `evaluation state`：仅用于评测恢复的附加状态

当前默认攻击目标为 `update_payload`，不是原始梯度。

具体含义如下：

- 若算法为普通稠密上传，攻击目标为方法对象导出的 dense update
- 若算法为 Top-K 稀疏上传，攻击目标先由 `result.sparse_update` 解压为稠密更新，并与 dense buffer update 合并
- 若算法为量化上传，攻击目标先由量化后的上传内容反量化为稠密更新
- 若算法为编码上传，攻击目标为方法对象显式导出的协议可见视图

因此，`update_payload` 模式的目标尽量对齐服务器在通信中实际看到的客户端上传内容，而不是简单读取某个固定字段。

框架也支持 `attack.target_type=gradient`：

- 此时不会直接用真实上传 payload 做攻击
- 而是从客户端额外采样一个 batch 的梯度作为攻击目标

当前正式实验主线主要使用 `update_payload`。

## 4. 攻击哪些客户端

被攻击客户端由 `attack.client_selection` 与 `attack.clients_per_round` 控制：

- `all`：攻击当前轮所有客户端
- `first`：只攻击前若干个客户端
- `round_robin`：按轮次轮换攻击客户端

当前如果实验配置设为 `all`，则每轮所有参与上传的客户端都会被攻击。

## 5. 每个客户端对应什么真实样本

### 5.1 `update_payload` 模式

对于被攻击客户端，服务器侧攻击任务会额外保存：

- `real_x, real_y`：从该客户端训练集按当前 DataLoader 顺序抽取的一个 batch
- `reference_inputs`：该客户端训练集可供参考匹配的一组输入窗口

这里的关键点是：

- 客户端实际上传的是“基于一个 epoch 本地训练后得到的更新”
- 但攻击评估时当前是拿该客户端训练集中的一个或若干个 batch 作为真实参考

因此它不是原始 DLG 论文里那种“单 batch 梯度泄露”的严格同构设定，而是更贴近“服务器拿到整轮 update 后，尝试还原该客户端训练数据分布中的样本”。

### 5.2 `gradient` 模式

对于 `gradient` 模式，服务器会调用客户端的梯度采样逻辑：

- 基于 `round_base_state`
- 从指定 batch 提取梯度
- 将该梯度作为攻击目标

这更接近原始 DLG / iDLG 的经典设定。

## 6. 压缩与保护如何进入攻击过程

### 6.1 `update_payload` 模式

在 `update_payload` 模式下，攻击直接使用客户端上传结果中可恢复出的 payload，因此已经天然包含了：

- 稀疏化
- 量化
- 反量化后的失真
- 其他上传侧处理后的影响

### 6.2 `gradient` 模式

在 `gradient` 模式下，代码会额外调用 `_protect_attack_gradients()`，对梯度施加与真实传输路径相对应的保护操作，例如：

- clipping
- noise
- top-k 稀疏化
- float16 / bfloat16 / int8 量化近似

因此，`gradient` 模式是“从梯度出发，再模拟通信保护”；`update_payload` 模式则是“直接攻击服务器实际截获到的上传内容”。

## 7. 服务器如何执行攻击

当前每个攻击样本都会分别运行两种方法：

- `DLG`
- `iDLG`

它们的共同目标都是优化一个虚拟输入 `dummy_x`，使其在当前模型上产生的目标尽量逼近服务器截获的目标。

### 7.1 DLG

- 同时优化 `dummy_x` 和 `dummy_y`

### 7.2 iDLG

- 仅优化 `dummy_x`
- `y` 固定为真实 `real_y`

### 7.3 目标函数

如果目标类型是 `gradient`：

- 比较 dummy batch 产生的梯度与截获梯度的距离

如果目标类型是 `update_payload`：

- 比较 dummy batch 产生的一步近似本地更新与截获更新的距离

其中 `update_payload` 模式里的本地更新不是完整重跑一个 epoch，而是采用一步近似：

- `SGD`: `-lr * grad`
- `Adam`: `-lr * grad / (|grad| + eps)`

所以当前 `update_payload` 攻击的本质是：

服务器试图找到一个 dummy 输入，使其诱导出“与真实上传 update 相似”的模型更新。

## 8. 攻击成功如何评估

当前实现中，每次攻击都会至少计算以下指标：

- `exact_target_mse`：重构输入与指定真实输入 `real_x` 的 MSE
- `psnr`
- `ssim`
- `gradient_mse`：攻击优化目标的残差

对于 `update_payload` 模式，还会额外计算：

- `nearest_client_train_mse`：重构输入与该客户端训练集参考窗口中最近邻样本的 MSE

## 9. 当前主指标是什么

主指标由 `attack.reference_metric` 控制。

默认规则如下：

- 对 `gradient` 攻击，默认主指标为 `reconstruction_mse`
- 对 `update_payload` 攻击，默认主指标为 `nearest_client_train_mse`

因此当前正式实验下，服务器更关注的是：

“重构结果是否已经足够接近该客户端训练集中的某个真实样本”

而不是要求它必须和某个指定 batch 完全一一对应。

## 10. 成功率的定义

每次攻击都会将主指标与固定阈值 `attack.success_mse_threshold` 做比较：

- 若 `primary_mse <= success_mse_threshold`，则记为一次成功攻击

随后再统计：

- 单方法成功率
- 总体成功率
- 平均 MSE
- 平均 PSNR
- 平均 SSIM
- 平均 objective / gradient MSE

## 11. 同步与异步攻击的区别

同步和异步的攻击内容相同，区别只在执行方式。

### 11.1 同步攻击

- 当前轮训练结束后立即执行攻击
- 主训练流程会等待攻击结束

### 11.2 异步攻击

- 当前轮只生成攻击任务快照
- 攻击任务提交给后台线程池执行
- 训练主流程继续向后推进
- 在整个训练结束前统一等待攻击任务全部完成

## 12. 异步攻击是否会污染训练和评测

当前实现中不会。

原因是攻击使用的是独立快照：

- 聚合前全局模型 `round_base_state` 的克隆
- 方法对象导出的 `attack view` 克隆
- `real_x / real_y / reference_inputs` 的克隆
- 如启用 `evaluation.mode=oracle_full_update`，评测使用的 `evaluation state` 与攻击视图分离

因此异步攻击不会修改聚合结果，也不会直接污染验证集或测试集性能。

异步攻击真正带来的影响主要是资源竞争，例如：

- GPU 算力竞争
- 显存占用
- wall-clock 时间变化

## 13. 当前攻击逻辑的一句话总结

当前框架中的服务器端攻击逻辑可以概括为：

服务器在每轮收到客户端上传后，基于聚合前的全局模型和各客户端实际上传的更新 payload，对选中的客户端分别运行 DLG / iDLG，重构可能的输入序列，并以重构结果和客户端训练数据之间的距离作为主要隐私泄露评估指标。

## 14. 相关代码位置

- 单节点联邦主循环：
  - `src/fedlab/federated/algorithms.py`
- 多节点 gRPC 联邦主循环：
  - `src/fedlab/communication/grpc_training.py`
- 攻击实现：
  - `src/fedlab/security/attacks.py`

