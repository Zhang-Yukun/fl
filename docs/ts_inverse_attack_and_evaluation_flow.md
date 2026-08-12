# TS-Inverse Attack And Evaluation Flow

本文档整理 **TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series Forecasting Models** 的攻击流程与评估流程，重点回答两个问题：

1. 它攻击的是什么参数或信号
2. 它用什么数据来评估攻击是否成功

本文档刻意区分两类信息：

- **已核实**：可直接从论文摘要页、论文主页或作者仓库公开说明确认
- **谨慎推断**：符合 GIA 一般范式，但当前公开摘要中未完整展开，后续应以论文正文/源码实现为准

## 1. 已核实的整体流程

从论文摘要页和作者仓库公开说明可以确认，TS-Inverse 属于 **gradient inversion attack (GIA)**，用于联邦时间序列预测场景。服务器通过客户端上传产生的梯度信息，尝试恢复原始时序数据，且恢复对象同时包含：

- `observations`
- `targets`

论文提出的 TS-Inverse 不是原样照搬 DLG / iDLG，而是在 GIA 框架上加入了三类时序专用增强：

1. 学习式的 gradient inversion model，输出 quantile predictions
2. 含 periodicity 与 trend regularization 的损失
3. 基于 quantile predictions 的进一步正则

## 2. 攻击的是什么

### 2.1 已核实部分

可以确认，TS-Inverse 攻击的是 **客户端训练过程中暴露给服务器的梯度信息**，而不是只基于最终预测结果做攻击。

公开材料里有两个直接信号：

- 论文标题和摘要都将 TS-Inverse 明确定义为 **gradient inversion attack**
- 作者仓库 README 明确提到需要把 **model parameters and gradients** 载入到相应变量中，说明攻击流程里显式使用了模型参数与梯度记录

因此可以确定：

- 攻击输入里包含模型参数状态
- 攻击核心目标里包含客户端上传或记录的梯度

### 2.2 当前不能写死的部分

就目前能直接核实的公开片段，还不能百分之百确认以下细节：

- 是对 **所有 trainable parameters** 的梯度做反演，还是只选取某一部分层
- 是否区分 backbone 参数与 head 参数
- 是否对某些层做裁剪、降维或筛选后再输入攻击模块

因此，当前更稳妥的表述是：

**TS-Inverse 攻击的是时间序列预测模型训练时暴露的梯度与对应模型参数状态，但“具体覆盖全部可训练参数还是某个参数子集”，仍建议以正文方法章节或公开实现为准。**

## 3. 它如何还原原始数据

### 3.1 已核实部分

从论文摘要可以确认，TS-Inverse 的恢复逻辑不是简单地从随机噪声开始做通用梯度匹配，而是结合了时序先验。

可确认的组成包括：

1. **Gradient inversion model**
   - 输入与梯度反演有关的信号
   - 输出 quantile predictions

2. **时序结构损失**
   - 包含 periodicity regularization
   - 包含 trend regularization

3. **基于 quantile predictions 的正则**
   - 用于约束最终恢复出来的时序结果

### 3.2 可以安全理解成的流程

用比较工程化的话讲，TS-Inverse 更像是下面这条链：

1. 服务器拿到客户端侧训练泄露出的梯度及对应模型参数状态
2. 利用一个学习式反演模型先给出“原始时序可能长什么样”的分位范围约束
3. 在恢复 observations 和 targets 的过程中，引入周期性与趋势正则
4. 通过这些时序先验把重构结果从“任意能匹配梯度的数值”拉回到“更像真实时序”的结果

## 4. 评估用什么数据

### 4.1 已核实部分

从论文摘要和仓库公开说明可以确认，TS-Inverse 评估时关注的是对真实时间序列中：

- `observations`
- `targets`

的恢复效果。

这说明评估不是只比较一个预测输出，也不是只判断攻击是否成功，而是直接比较：

- 重构出的 observation 与真实 observation
- 重构出的 target 与真实 target

### 4.2 评估数据的来源

可以较确定地说，评估使用的是**被攻击样本对应的真实时间序列数据**，因为论文目标就是衡量恢复出的 observations / targets 与原始数据有多接近。

更直白地说：

- 攻击输出：恢复得到的时序 observation / target
- 评估基准：该次被攻击训练样本的真实 observation / target

### 4.3 当前不能完全写死的部分

目前公开摘要没有完全展开以下细节，因此不宜在复现文档里写成“已确认事实”：

- observation 与 target 是分别评分还是联合评分
- 是否按 sample 先求分数再平均
- 是否按时间步先聚合再对 batch 平均
- 是否同时报告变量维度上的分项结果

这些更细的聚合方式仍建议以正文评估章节为准。

## 5. 评估指标是什么

### 5.1 已核实部分

论文和仓库公开说明都表明，TS-Inverse 的主评估指标是 **sMAPE**。

因此，它的核心比较方式是：

- 重构序列与真实序列之间的对称相对误差
- 指标越小，说明攻击越强、恢复越准确

### 5.2 评估对象

结合论文公开描述，sMAPE 评估对象至少覆盖：

- 被重构的 observations
- 被重构的 targets

所以论文的主结论并不是“成功率多少”，而是“在 observations / targets 的时序重构上，sMAPE 是否显著优于其他 GIA 方法”。

## 6. 可以整理成的攻击-评估闭环

基于目前已核实信息，TS-Inverse 的完整闭环可以整理成下面这样：

1. 客户端在时间序列预测任务上训练模型
2. 训练过程中产生模型梯度，并被服务器端攻击者利用
3. 服务器侧攻击模块结合：
   - 模型参数状态
   - 梯度信息
   - 学习式 quantile prediction
   - periodicity / trend regularization
4. 输出重构的 observations 与 targets
5. 用真实 observations 与真实 targets 作为 ground truth
6. 用 sMAPE 衡量恢复误差
7. 与既有 GIA 方法做对比

## 7. 和 DLG / iDLG 的主要差别

和 DLG / iDLG 相比，TS-Inverse 的重点不只是“让 dummy 输入匹配梯度”，而是进一步回答：

- 时序数据有周期性和趋势，如何把这些结构先验用进反演
- 时序预测任务不仅有输入 observation，还有未来 target，如何一起恢复
- 评价时不只看图像式 MSE/PSNR，而是用更适合时序预测的 sMAPE

## 8. 当前最稳妥的结论

截至目前，可以较确定地写出下面三点：

1. **攻击信号**：TS-Inverse 攻击的是联邦时间序列预测中暴露给服务器的梯度信息，并结合模型参数状态进行重构。
2. **恢复对象**：恢复的不是抽象特征，而是原始时间序列中的 `observations` 和 `targets`。
3. **评估数据与指标**：评估时使用被攻击样本对应的真实 observation / target 作为基准，主指标为 `sMAPE`。

而下面这些点目前不建议写死：

- 是否使用全部可训练参数的梯度
- 是否对不同层做了筛选
- sMAPE 在 observation / target / batch / 时间维上的最终聚合顺序

## 9. 参考来源

- 论文主页：`arXiv:2503.20952`
- 代码仓库：`Capsar/ts-inverse`

