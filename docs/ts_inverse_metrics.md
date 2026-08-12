# TS-Inverse Evaluation Metrics

本文档整理论文 **TS-Inverse: A Gradient Inversion Attack Tailored for Federated Time Series Forecasting Models** 使用的攻击评估指标。

本文档目标是帮助后续做指标对齐与实验复现。内容分为两部分：

- 已核实：可以直接从论文摘要页、论文主页或作者仓库公开说明中确认
- 待正文确认：当前公开摘要中能看出方向，但具体公式或聚合方式仍建议回到论文正文对应小节逐项核对

## 1. 结论概览

从论文摘要页和作者仓库公开说明可以确认，TS-Inverse 的主评估指标是 **sMAPE**，并且论文将结果表述为：

- 与现有 GIA 方法相比，TS-Inverse 在时序数据上的 **sMAPE** 有显著改善
- 论文强调其攻击对象同时包含 **observations** 和 **targets**

因此，TS-Inverse 的评价思路与图像场景常见的 `PSNR / SSIM / success rate` 不同，更偏向于：

- 把重构序列当作时序预测值
- 用时序误差指标衡量“重构得有多接近真实序列”

## 2. 已核实的指标信息

### 2.1 主指标是 sMAPE

目前可以明确确认：

- TS-Inverse 使用 **sMAPE** 作为核心攻击效果指标
- 论文摘要与仓库说明都直接用 sMAPE 描述主实验结论

这意味着：

- 指标越小，表示重构越准确
- 指标越大，表示攻击效果越差、隐私泄露越弱

### 2.2 指标针对的是时序值重构

论文摘要中明确说明，TS-Inverse 关注的是对时间序列中的：

- `observations`
- `targets`

的重构效果。

因此它不是只看输入窗口，也不是只看预测目标，而是明确把两部分都作为攻击对象来讨论。

### 2.3 主结果表述是误差改善，不是二值成功率

从目前能直接核实的论文摘要和仓库说明看，TS-Inverse 主结果的表达方式是：

- “在 sMAPE 上优于现有方法多少”

而不是：

- “攻击成功率达到多少”

这说明论文主视角是连续误差型评估，而不是阈值化成功/失败评估。

## 3. sMAPE 的常见定义

TS-Inverse 公开说明没有在摘要里直接展开公式，但它提到的 sMAPE 一般按下面形式定义：

\[
\mathrm{sMAPE}(y, \hat{y}) =
\frac{1}{N}\sum_{i=1}^{N}
\frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \varepsilon}
\]

其中：

- \( y_i \) 为真实值
- \( \hat{y}_i \) 为重构值
- \( \varepsilon \) 用于避免分母为 0

有些实现会再乘 `100%`，有些不会。无论是否乘 `100`，比较不同方法时只要实现一致即可。

## 4. 与图像场景常见 GIA 指标的区别

TS-Inverse 的评价方式和很多图像类梯度反演论文存在明显差别。

图像场景常见的指标包括：

- `MSE`
- `PSNR`
- `SSIM`
- `attack success rate`

而 TS-Inverse 的公开主结果更强调：

- `sMAPE`

这样做的原因很直接：

- 时序预测任务本身就是回归任务
- 时序序列更适合用连续误差指标衡量数值偏差
- `PSNR / SSIM` 在时序上不是最自然的主指标

## 5. 可以较大概率推断但仍建议正文确认的部分

下面这些点目前从摘要可以看出方向，但还不建议在复现中直接视为“百分之百已确认”。

### 5.1 是否分别报告 observation 与 target 的误差

论文摘要明确说攻击的是 observations 和 targets，但当前公开摘要片段没有完全展开：

- 是否分别报告 `obs_sMAPE`
- 是否分别报告 `target_sMAPE`
- 是否还报告一个联合指标

这部分需要回到论文正文的评估章节确认。

### 5.2 是否对 batch 内样本、时间维、变量维做了特定聚合

对于时序数据，sMAPE 可能存在几种常见聚合方式：

- 先对单样本、单时间点计算，再全局平均
- 先对每条序列求 sMAPE，再对 batch 平均
- 对 observation 与 target 分别平均后再汇总

当前公开摘要没有把这一步聚合方式讲清楚，因此后续对齐时需要以正文公式或作者实现为准。

### 5.3 是否还报告辅助指标

从论文风格判断，作者很可能还会给出一些辅助型定性或定量指标，但目前能直接核实为“主指标”的只有 sMAPE。

是否额外报告如下指标，当前不能仅凭摘要直接下结论：

- `MAE`
- `MSE`
- `RMSE`
- `DTW`
- `success rate`

因此在没有正文佐证前，不建议把这些写成 TS-Inverse 的标准主指标。

## 6. 对复现实验最重要的启示

如果希望实验口径更接近 TS-Inverse，至少应当遵循下面几个原则：

1. 主指标优先使用 **sMAPE**
2. 指标对象应覆盖 **time-series observations** 与 **forecasting targets**
3. 主比较方式优先使用 **连续误差对比**
4. 不应只依赖阈值化成功率作为唯一结论

## 7. 建议的对齐拆分方式

如果后续需要参考 TS-Inverse 做指标设计，一个稳妥的拆分方案是：

- `obs_smape`：重构 observation 与真实 observation 的 sMAPE
- `target_smape`：重构 target 与真实 target 的 sMAPE
- `joint_smape`：将两部分统一聚合后的 sMAPE

这样做的好处是：

- 能覆盖论文强调的 observation + target 双重重构问题
- 即便后续确认论文正文采用的是分开报告或联合报告，也都容易兼容

## 8. 当前可确认与不可确认边界

截至目前，可以较确定地说：

- TS-Inverse 的核心评估指标是 **sMAPE**
- 其攻击目标包含 **observations** 和 **targets**
- 论文主结论是基于 **sMAPE 改善幅度** 展开的

但当前不宜写死的内容包括：

- observation 与 target 是否分开打分
- 最终是否使用联合 sMAPE 作为唯一主表格指标
- sMAPE 在 batch / 时间维 / 通道维上的最终聚合顺序

这些细节后续应以论文正文评估章节或作者公开实现为准。

## 9. 参考来源

- arXiv 论文主页：`arXiv:2503.20952`
- 作者仓库：`Capsar/ts-inverse`

