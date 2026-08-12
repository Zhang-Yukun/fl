# TS-Inverse Source Code Notes

本文档基于 `Capsar/ts-inverse` 源码整理 TS-Inverse 的攻击流程、训练流程与评估细节。这里重点写**代码里实际发生了什么**，而不是只复述论文摘要。

## 1. 代码入口与角色划分

和攻击最相关的几个文件是：

- `src/ts_inverse/workers/attack_dlg_invg_dia_worker.py`
  - 基线攻击实现，包含 DLG / INVG / DIA 一类的优化式攻击基座
- `src/ts_inverse/workers/attack_learning_to_invert_worker.py`
  - 学习式 inversion model 的训练与推理
- `src/ts_inverse/workers/attack_ts_inverse_worker.py`
  - TS-Inverse 主实现，在学习式 inversion model 基础上叠加时序正则和 quantile 约束
- `src/ts_inverse/workers/attack_worker.py`
  - 通用攻击工具，包括梯度记录、dummy 初始化、评价与日志
- `src/ts_inverse/attack_time_series_utils.py`
  - `SMAPELoss`、`pinball_loss`、trend / periodicity regularization
- `src/ts_inverse/models/grad_to_input.py`
  - 从梯度映射回输入/目标的 inversion model 结构

## 2. 攻击的输入到底是什么

### 2.1 TS-Inverse 攻击的是梯度，不是模型更新

代码里攻击的直接输入是 `original_dy_dx`，它来自：

- `AttackBaselineWorker.start_attack()`
- 然后调用 `self.attack_batch(..., original_dy_dx, ...)`

这个 `original_dy_dx` 来自 `train_model_and_record()` 中保存的：

- `all_model_gradients.append([param.grad.clone() for param in model.parameters()])`

也就是说，**TS-Inverse 实际攻击的是每个 batch 反向传播后得到的逐参数梯度列表**，不是 FedAvg 那种整轮模型 update。

### 2.2 梯度是在哪个时刻记录的

在 `train_model_and_record()` 中，处理顺序是：

1. `model_optimizer.zero_grad()`
2. `out = model(batch_inputs)`
3. `y = F.mse_loss(out, batch_targets)`
4. `y.backward()`
5. 记录 `param.grad`
6. `model_optimizer.step()`

因此记录的是：

- **当前 batch**
- **当前模型参数状态下**
- **MSE 损失回传之后**

的原始梯度。

### 2.3 它记录了哪些参数

代码使用的是：

- `for param in model.parameters()`
- `torch.autograd.grad(..., model.parameters(), ...)`

因此，默认情况下是**对模型所有可训练参数的梯度逐层记录并用于攻击**，不是只取最后一层。

不过，损失函数里可以人为只看最后两层，例如：

- `gradient_loss == "last_2_layers_l1"`

这表示：

- 攻击目标梯度默认是全参数
- 但优化时可以只对其中一部分梯度求匹配损失

## 3. 攻击样本是什么

### 3.1 输入与目标是怎么取的

在 `train_model_and_record()` 里，训练 batch 被处理成：

- `batch_inputs = batch_inputs[:, :, model.features]`
- `batch_targets = batch_targets[:, :, 0]`

这说明：

- 输入序列只取模型声明的 `features`
- 目标序列只取 target tensor 的第 `0` 个特征

因此 TS-Inverse 实际恢复的是：

- observation window 中被选定的输入特征
- forecasting target 中第一个目标变量

### 3.2 攻击单位是 batch

代码是按 `batch_number` 逐个 batch 攻击的。  
`AttackBaselineWorker.start_attack()` 会对每个被记录的 batch：

1. 取出该 batch 的真实输入和真实目标
2. 取出该 batch 的梯度
3. 初始化对应的 dummy inputs / dummy targets
4. 运行攻击

因此它不是“从整个客户端数据集反演”，而是**从单个训练 batch 的平均梯度反演该 batch 的样本**。

## 4. 模型训练与梯度记录时的优化器

源码里梯度记录阶段固定使用：

- `torch.optim.SGD(model.parameters(), lr=0.001)`

这一点很关键，因为它意味着：

- 攻击实验的梯度来源是 SGD 训练过程
- 不是 Adam，也不是论文摘要里抽象的“任意训练器”

## 5. 是否支持防御后再攻击

支持，而且是直接在记录梯度时做。

在 `train_model_and_record()` 以及 `create_gradient_inversion_dataloader()` 中，如果配置里有 `defense_name`，则可以对梯度做：

- `sign`：符号化
- `prune_rate`：稀疏剪枝
- `dp_epsilon`：加高斯噪声

然后再把被防御后的梯度：

- 作为攻击目标梯度
- 以及作为 inversion model 的训练数据输入

也就是说，**TS-Inverse 可以直接在“被防御后的梯度视图”上训练和攻击**。

## 6. TS-Inverse 和基线攻击的关系

`AttackTSInverseWorker` 继承自 `AttackLearningToInvertWorker`，而后者又继承自 `AttackBaselineWorker`。

这意味着 TS-Inverse 的结构是三层叠加：

1. 基线优化式梯度匹配攻击
2. 学习式 gradient-to-input inversion model
3. 时序专用 regularization 与 quantile prior

所以 TS-Inverse 不是完全替代 DLG 一类方法，而是在同一套优化骨架上继续增强。

## 7. 学习式 inversion model 是怎么训练的

### 7.1 它需要一个辅助数据集

`AttackLearningToInvertWorker.init_attack()` 会构造：

- `auxiliary_train_dataloader`
- `auxiliary_val_dataloader`

来源有两种：

1. 如果配置了 `aux_dataset`，就用单独指定的辅助数据集
2. 否则默认直接用当前任务的：
   - `test_datasets` 作为 auxiliary train
   - `val_datasets` 作为 auxiliary val

这点很重要：  
**代码默认的辅助先验数据并不一定是完全外部数据，它可能直接来自当前任务的验证/测试切分。**

### 7.2 梯度到样本的数据集怎么做

`create_gradient_inversion_dataloader()` 会遍历辅助数据集，针对每个辅助样本：

1. 跑一次模型前向和 MSE loss
2. 反向传播得到梯度
3. 将所有参数梯度 flatten 后拼接成一个长向量
4. 保存三元组：
   - `flattened_gradient`
   - `aux_inputs`
   - `aux_targets`

最后组成：

- `TensorDataset(aux_dy_dx_inputs, aux_inputs_targets, aux_targets_targets)`

这就是 inversion model 的监督训练数据。

### 7.3 inversion model 学的是什么映射

它学的是：

- 输入：flatten 后的梯度向量
- 输出：
  - observation 序列
  - target 序列

也就是典型的：

`gradient -> (input sequence, target sequence)`

## 8. inversion model 的结构

源码里有多种 gradient-to-input 网络：

- `GradToInputNN`
- `ImprovedGradToInputNN`
- `ImprovedGradToInputNN_2`
- `ImprovedGradToInputNN_Quantile`
- `ImprovedGradToInputNN_Probabilistic`

### 8.1 普通版本

普通版本基本都是：

- 输入一个 flatten gradient 向量
- 经过 MLP / residual MLP
- 分成两个 head：
  - 一个输出 input reconstruction
  - 一个输出 target reconstruction

### 8.2 Quantile 版本

`ImprovedGradToInputNN_Quantile` 的输出不是一个点估计，而是：

- 对每个输入位置输出多个 quantiles
- 对每个 target 位置输出多个 quantiles

形状上，最后多出一维 `len(quantiles)`。

### 8.3 Probabilistic 版本

`ImprovedGradToInputNN_Probabilistic` 会输出分布参数，而不是直接点值。

支持三种分布：

- `normal`
- `cauchy`
- `beta`

然后在 `inference()` 时从该分布采样出 inputs / targets。

## 9. inversion model 的训练损失

### 9.1 MSE 模式

普通点预测模式下，inversion model 的 loss 不是简单逐样本顺序对齐，而是：

1. 先把 predicted inputs / targets 与 auxiliary inputs / targets flatten
2. 构造 batch 内样本之间的 pairwise distance matrix
3. 用 `linear_sum_assignment` 做最优匹配
4. 对匹配后的样本损失求平均

这表示训练时对 batch 内样本顺序是**置换不敏感**的。

### 9.2 Quantile 模式

如果 `attack_loss == "quantile"`，则训练时使用：

- `pinball_loss`

对 quantile prediction 进行监督。

当 `inversion_batch_size != batch_size` 时，会把预测和真实 flatten 后直接做整体 `pinball_loss`。  
否则会构造 batch-wise cost matrix，再做 `linear_sum_assignment`。

### 9.3 Probabilistic 模式

如果 `attack_loss` 是：

- `normal`
- `cauchy`
- `beta`

则损失是对应分布的负对数似然。

## 10. TS-Inverse 主攻击流程

### 10.1 先训练或加载 inversion model

`AttackTSInverseWorker.attack_batch()` 首先会调用父类方法：

- 训练 inversion model，或
- 从 `../data/_model_dataset_gradients/` 加载已有模型

### 10.2 再进行优化式重构

随后 TS-Inverse 会进入真正的梯度匹配优化阶段：

1. 取当前被攻击 batch 的真实梯度 `original_dy_dx`
2. 用 inversion model 给出先验预测
3. 初始化或正则 dummy inputs / dummy targets
4. 以梯度差损失为主目标做迭代优化

### 10.3 攻击优化目标

在每一步里：

1. `dummy_out = model(dummy_inputs)`
2. `dummy_y = F.mse_loss(dummy_out, dummy_targets)`
3. `dummy_dy_dx = autograd.grad(dummy_y, model.parameters(), create_graph=True)`
4. `dy_dx_loss += gradient_loss_function(dummy_dy_dx, original_dy_dx, ...)`

所以核心仍然是：

**让 dummy 数据产生的梯度尽量接近真实梯度**

这点和 DLG / INVG 一致。

## 11. TS-Inverse 额外加了哪些正则

TS-Inverse 比基线多出来的部分主要在 `attack_ts_inverse_worker.py`。

### 11.1 Learned prior regularization

如果有 inversion model，就可以把它的输出作为先验。

支持三类 regularization：

1. `quantile`
   - 用 quantile prediction 对 dummy 序列施加 pinball loss
2. `quantile_bounds`
   - 惩罚 dummy 序列落在预测 quantile 区间之外
3. `l1`
   - 直接把 dummy 序列拉向 inversion model 的点预测

### 11.2 Trend regularization

使用 `trend_consistency_regularization()`：

- 先对序列做线性趋势拟合
- 再约束序列与其线性趋势之间的偏差

可以选：

- `l1_mean`
- `l1_sum`
- `l2_mean`
- `l2_sum`

### 11.3 Periodicity regularization

使用 `periodicity_regularization(sequence, period, loss=...)`：

- 对一个点和“一个周期之后”的对应点做一致性约束

代码里 `period` 直接取：

- `dummy_targets.shape[1]`

而且 regularization 用的是：

- `combined_dummy_data_first_feature = cat([dummy_inputs[:, :, 0], dummy_targets], dim=1)`

也就是说：

- 它只对**输入的第一个特征 + 整段 targets 拼接后的同一条序列**做趋势/周期正则
- 不是对全部输入特征逐个做 trend / periodicity

这是源码里一个很重要的细节。

### 11.4 Total variation regularization

还支持：

- `total_variation_alpha_inputs`
- `total_variation_beta_targets`

本质上约束时间上相邻点变化不要过于剧烈。

### 11.5 Lower resolution regularization

还支持：

- `lower_res_term_inputs`
- `lower_res_term_targets`

做法是：

1. 先把序列做 temporal resolution warping
2. 再插值恢复到原长度
3. 约束恢复后的低分辨率平滑版本与当前 dummy 序列接近

这相当于鼓励重构结果在低频尺度上更平滑。

## 12. 梯度损失有哪些形式

`gradient_loss_function()` 支持很多损失：

- `l1`
- `euclidean`
- `cosine_invg`
- `cosine_dia`
- `last_2_layers_l1`
- `top20percent_l1`
- `double_outer_l1`
- 以及 norm + cosine 的混合形式

因此，TS-Inverse 并不是绑定某一种梯度相似度，而是允许在多种匹配损失上做实验。

## 13. 特殊技巧

### 13.1 one-shot target recovery

如果：

- `batch_size == 1`
- 且 `one_shot_targets == True`

代码会从最后两层梯度 `grad_w` 和 `grad_b` 直接构造一个 target 近似值，而不是完全靠迭代优化。

这属于一个非常特定的单样本技巧。

### 13.2 dropout 一起优化

如果模型是 TCN，且开启：

- `optimize_dropout`

那么攻击时不仅优化 dummy inputs / targets，还会把 dropout mask 也纳入优化变量。

这说明作者显式考虑了带 dropout 的攻击难点。

### 13.3 gradient sign

攻击阶段可以对 `dummy_inputs.grad`、`dummy_targets.grad` 甚至 dropout mask 的梯度取 sign，再进行更新。

## 14. 评估是怎么做的

### 14.1 对象

评估对象分成两类：

- `inputs`
- `targets`

也就是 observation reconstruction 和 target reconstruction 分开算。

### 14.2 指标

`evaluate_and_log_reconstruction()` 里对 inputs 和 targets 都计算：

- `mse`
- `rmse`
- `mae`
- `smape`

包括：

- batch 平均指标，例如 `inputs/smape/mean`
- 每个样本单独指标，例如 `targets/mse/0`

所以从源码看，**TS-Inverse 并不是只看 sMAPE**；  
只是论文主结果可能更强调 sMAPE。

### 14.3 sMAPE 的实现

`SMAPELoss(y, y_hat)` 定义为：

\[
2|y-\hat y| / (|y| + |\hat y|)
\]

逐点平均。

代码里没有乘 `100`，因此它报告的是比例值，而不是百分数形式。

### 14.4 样本对齐方式

评估时如果 batch size > 1，代码不会强制按原顺序逐个对齐，而是先做 sample mapping。

`get_batch_sample_mapping()` 的逻辑是：

1. 对每个真实样本，找 loss 最小的 dummy 样本
2. 默认用 `L1` 做这个匹配
3. 如果匹配结果出现重复索引，就直接退回 identity mapping

这点很关键：

- **训练 inversion model 时**，他们用的是 Hungarian assignment
- **最终攻击评估时**，他们用的是贪心最近邻匹配，如果出现重复就回退到原顺序

这两个阶段的匹配策略并不一样。

### 14.5 Quantile 模式下的评估

如果 `attack_loss == "quantile"`，那么 `evaluate_dummy_prediction()` 会专门改成：

- 对 quantile outputs 用 `pinball_loss` 评估
- 记录 `inputs/pinball/mean`、`targets/pinball/mean`

并画出每个 quantile 的重构曲线。

注意：

- 这一部分是对 inversion model 输出本身的评估
- 真正最终的 dummy reconstruction 仍会回到 `evaluate_and_log_reconstruction()` 里按 `mse/rmse/mae/smape` 打分

## 15. 论文摘要和源码之间最值得注意的几个差异

### 15.1 论文主叙事强调 sMAPE，但源码实际记录了更多指标

源码明确同时记录：

- MSE
- RMSE
- MAE
- sMAPE

所以如果只看论文摘要，很容易误以为它只用 sMAPE。

### 15.2 论文说恢复 observations 和 targets，源码里确实是分开恢复、分开评分

这一点在代码中很明确，不是只恢复 observation 或只恢复 target。

### 15.3 摘要没有强调的一点：辅助 inversion model 默认可能直接用 val/test 数据构造

这在源码里是非常明显的实现选择。

### 15.4 论文层面容易忽略的一点：trend / periodicity 正则主要只作用在第一输入特征和 target 拼接后的单条序列上

不是所有输入通道都对称地施加了这个约束。

## 16. 对你们框架最有参考价值的结论

如果要借鉴 TS-Inverse，而不是只借用它的名字，那么至少应把下面几点分开看：

1. **攻击信号层面**
   - 它攻击的是 batch 梯度，不是整轮联邦 update

2. **先验学习层面**
   - 它先训练了一个 `gradient -> input/target` inversion model

3. **优化层面**
   - 它仍然保留了 DLG 类梯度匹配优化

4. **时序正则层面**
   - trend / periodicity / lower-resolution / TV 都是附加约束

5. **评估层面**
   - 它实际同时记录 `MSE / RMSE / MAE / sMAPE`
   - observation 和 target 是分开评分的

