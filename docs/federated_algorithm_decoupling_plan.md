# 联邦算法解耦方案

## 目标

当前联邦算法的客户端上传、服务器聚合、攻击视图提取、是否压缩等语义散落在：

- `fedlab/federated/client.py`
- `fedlab/federated/server.py`
- `fedlab/federated/algorithms.py`
- `fedlab/communication/grpc_training.py`

这会导致新增算法时需要在多个文件中同步修改同一份语义。目标是把算法收敛到统一接口下，并让运行时只依赖注册表和抽象接口，而不是硬编码字符串分支。

## 目标结构

计划引入新的算法模块层：

- `fedlab/federated/methods/base.py`
  - 联邦算法基类
  - 算法能力描述（是否压缩、说明等）
- `fedlab/federated/methods/registry.py`
  - 算法注册表
  - 配置到算法实例的解析逻辑
- `fedlab/federated/methods/`
  - 每个算法族一个模块
  - 例如 `dense.py`、`sparse.py`、`quantized.py`、`encoded.py`

## 统一接口

每个联邦算法最终需要显式实现以下职责：

1. `configure_client(client)`
   - 客户端初始化时的附加状态
   - 例如 EGA 编码器
2. `configure_server(server)`
   - 服务器初始化时的附加状态
   - 例如隐私记账器、编码器
3. `build_round_context(server)`
   - 每轮由服务器下发的算法上下文
4. `client_update(...)`
   - 客户端如何从本地训练结果构建上传 payload
5. `aggregate(...)`
   - 服务器如何从客户端 payload 聚合全局更新
6. `extract_attack_payload(...)`
   - 服务器攻击端真实可见的 payload 视图

## 关键语义拆分

后续所有算法都要围绕以下三层语义组织，而不是把它们混成一个 `result.state`：

1. `aggregation payload`
   - 用于真正聚合的上传内容
2. `attack view`
   - 服务器攻击时真实能看到的内容
3. `evaluation state`
   - 仅用于验证/测试时需要恢复的模型状态

这三层语义允许压缩、攻击、性能评估在逻辑上显式区分，减少异步攻击、多节点运行时的混淆。

## 迁移顺序

为了降低风险，按以下顺序迁移：

1. 定义接口、注册表、算法名册，不改变现有行为
2. 让运行时通过注册表获取算法对象，但先保留现有逻辑
3. 先迁移 dense 算法族
   - `fedavg`
   - `fedaware`
   - `adaptive_clipped_rdp_fedavg`
4. 再迁移 sparse/quantized 算法族
   - `compressed_fedavg`
   - `sparse_fedavg`
   - `dp_topk_fedavg`
   - `randomk_fedavg`
   - `soteriafl`
   - `secure_quantized_fedavg`
   - `sign_fedavg`
   - `qsgd_fedavg`
5. 最后迁移特殊算法
   - `ega_fedavg`
6. 清理旧的字符串分支和过渡兼容逻辑

## 每阶段验证要求

每个关键阶段都要满足：

- 单节点 `pytest` 通过
- gRPC 相关回归测试通过
- 配置一致性测试通过
- 关键算法 smoke test 能跑通
- 完成后单独 git 提交

## 文档更新要求

重构完成后需要同步更新：

- `README.md`
- `docs/框架使用手册.md`
- 算法接入说明
- 攻击/评估语义说明
