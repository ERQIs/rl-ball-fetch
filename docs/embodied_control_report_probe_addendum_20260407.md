# Embodied Control Report Addendum: RL Post-Probe Results

本补充文档用于追加主报告 `embodied_control_report_20260406.md` 中尚未展开的 RL 后表征 probe 结果，供毕设正文或附录直接引用。

## 1. Probe 对象与数据

probe 使用的强化学习后 checkpoint 为：

- `rl_be/results/exp_ec_0001/temp_realTime_0001/CarCatch/checkpoint.pt`

probe 数据仍然来自离线采集轨迹数据集：

- `ball_fetch/vis_backbone/datasets/manual_capture/20260401_101517`

本次 full probe 使用：

- 训练 episode：`800`
- 测试 episode：`100`
- 训练窗口数：`26812`
- 测试窗口数：`3354`

## 2. Probe 表征与目标

分析的内部表征包括：

- `short`：短时视觉表征 `z_short`
- `self`：自身状态表征 `z_self`
- `memory`：融合后的记忆输入 `z_mem`
- `long`：长时记忆状态 `h_long`
- `readout`：动作头读出前的融合表征

线性 probe 的目标包括：

- `landing_offset`
- `ball_rel_pos`
- `ball_rel_vel`
- `self_velocity`
- `time_to_land`

## 3. Full Probe 结果

测试集 `R^2` 结果如下：

| feature | landing_offset | ball_rel_pos | ball_rel_vel | self_velocity | time_to_land |
|---|---:|---:|---:|---:|---:|
| short | 0.6154 | 0.9964 | 0.5644 | 0.6406 | 0.9806 |
| self | 0.1833 | 0.6190 | 0.4658 | 1.0000 | 0.6295 |
| memory | 0.7093 | 0.9905 | 0.7953 | 0.9777 | 0.9793 |
| long | 0.6879 | 0.9831 | 0.7900 | 0.9282 | 0.9763 |
| readout | 0.7753 | 0.9975 | 0.8650 | 1.0000 | 0.9880 |

## 4. 结果解释

### 4.1 `short` 更像当前空间知觉

`short` 对 `ball_rel_pos` 的 `R^2 = 0.9964`，对 `time_to_land` 的 `R^2 = 0.9806`。这说明短时视觉表征中已经几乎显式编码了：

- 球相对车当前在哪里
- 球距离落地还有多久

因此，`z_short` 很像“当前场景几何与当前运动状态”的短时空间知觉表示。

### 4.2 `self` 基本是纯本体感觉

`self` 对 `self_velocity` 的 `R^2 = 0.999992`，接近完美。这说明 `z_self` 的主要职责确实是编码“我自己现在怎么动”，而不是偷偷承载外界球状态。

### 4.3 `memory` 和 `long` 更像预测性轨迹记忆

与 `short` 相比，`memory` 和 `long` 在以下目标上更强：

- `landing_offset`
- `ball_rel_vel`

其中：

- `memory` 对 `landing_offset` 的 `R^2 = 0.7093`
- `memory` 对 `ball_rel_vel` 的 `R^2 = 0.7953`
- `long` 对 `landing_offset` 的 `R^2 = 0.6879`
- `long` 对 `ball_rel_vel` 的 `R^2 = 0.7900`

这说明长时分支不只是“记住过去看过什么”，而是在累积对球飞行趋势和未来结果的判断，因此更接近一种预测性的具身记忆。

### 4.4 `readout` 是面向动作的综合表征

`readout` 在几乎所有 probe 目标上都是最强的：

- `landing_offset`: `0.7753`
- `ball_rel_pos`: `0.9975`
- `ball_rel_vel`: `0.8650`
- `self_velocity`: `1.0000`
- `time_to_land`: `0.9880`

这说明动作头读出前的融合表示，确实把三类关键信息整合在了一起：

- 当前视觉空间信息
- 长时轨迹记忆
- 当前自身速度信息

## 5. 可写入正文的结论表述

基于上述 probe，可以较为谨慎地给出如下结论：

1. 该模型并非只是在做动作拟合，而是学到了任务相关的空间表征。
2. 该空间表征是以自身为中心的（egocentric），而不是依赖球的 privileged state 输入。
3. 该空间表征具有明显的预测性，因为长时分支能够更好地解码落点偏移和相对速度。
4. 最终动作不是单独依赖瞬时视觉或单独依赖记忆，而是建立在短时视觉、本体感觉和长时记忆三者融合的基础上。

因此，在论文中更稳妥的说法不是“学到了完整世界模型”，而是：

- 学到了 `task-oriented spatial cognition`
- 学到了 `egocentric predictive representation`
- 学到了 `predictive sensorimotor representation`

## 6. 下一步建议

为了进一步加强这个结论，下一步最值得做的是：

1. 对 warmup 模型做同样的 full probe，与 RL 后模型逐项对比。
2. 对 `short`、`self`、`long` 分别做因果遮挡或置零实验，直接看动作变化。
3. 做跨随机种子或跨发球分布的 probe 稳定性验证。

## 7. Warmup 与 RL 后模型的对照分析

为了判断强化学习阶段到底“新学到了什么”，还是“只是沿用了 warmup 的表征”，本课题进一步对 warmup checkpoint 与 RL 后 checkpoint 做了同样的 full probe 对照。

warmup 使用的 checkpoint 为：

- `temperal/experiments/embodied_control/output/offline_warmup_manual_capture_20260401_101517/best.pt`

RL 后使用的 checkpoint 为：

- `rl_be/results/exp_ec_0001/temp_realTime_0001/CarCatch/checkpoint.pt`

### 7.1 关键 `R^2` 对比

| feature | target | warmup | RL后 |
|---|---|---:|---:|
| short | landing_offset | 0.7420 | 0.6154 |
| short | ball_rel_pos | 0.9976 | 0.9964 |
| short | ball_rel_vel | 0.6116 | 0.5644 |
| short | time_to_land | 0.9867 | 0.9806 |
| memory | landing_offset | 0.8183 | 0.7093 |
| memory | ball_rel_vel | 0.8419 | 0.7953 |
| long | landing_offset | 0.8987 | 0.6879 |
| long | ball_rel_vel | 0.8705 | 0.7900 |
| readout | landing_offset | 0.9014 | 0.7753 |
| readout | ball_rel_pos | 0.9985 | 0.9975 |
| readout | ball_rel_vel | 0.9087 | 0.8650 |
| readout | time_to_land | 0.9929 | 0.9880 |
| self | self_velocity | 1.0000 | 1.0000 |
| readout | self_velocity | 1.0000 | 1.0000 |

### 7.2 对比结论

从这组结果可以看到，一个非常稳定的趋势是：

1. warmup 几乎在所有 probe 指标上都强于 RL 后模型。
2. 这种优势在 `landing_offset` 和 `ball_rel_vel` 上尤其明显。
3. 即便如此，RL 后模型在 `ball_rel_pos`、`time_to_land`、`self_velocity` 等关键量上仍然保持了很高的线性可解码性。

这说明：

- warmup 阶段学到的是更“显式”的预测性表征。
- RL 阶段并没有把这些表征学没，而是把它们压缩并重组为更面向动作控制的内部状态。

### 7.3 如何理解这种变化

warmup 的训练目标本身就是监督式的未来理解任务，因此模型会倾向于把：

- 落点偏移
- 剩余时间
- 相对速度

这些量显式写入隐藏状态，使得它们更容易被线性 probe 直接读出来。

而在 RL 阶段，优化目标不再是“把未来变量表示得尽可能清楚”，而是“把球接到”。因此，策略网络只需要保留那些真正有助于控制决策的信息，而不必维持对所有预测量都同样显式、同样线性的编码方式。

因此，RL 后 probe 变弱，不应简单理解为“空间认知消失了”，更合理的解释是：

- 表征从“监督学习友好”转向了“控制决策友好”
- 表征仍然保留了关键空间信息，但形式变得更紧凑、更任务导向

### 7.4 可写入正文的总结表述

基于 warmup 与 RL 后模型的对照，本课题可以给出更完整的解释：

1. warmup 阶段先塑造了较强的预测性时空表征，为后续控制提供先验。
2. RL 阶段不是从零学习空间认知，而是在已有表征的基础上进行任务适配。
3. 适配后的表征虽然在线性 probe 上略弱于 warmup，但仍然稳定保留了接球任务所需的关键空间信息。
4. 因此，最终 agent 的成功并不是偶然动作拟合，而是建立在“先学会理解轨迹，再把理解重构为动作策略”的两阶段学习过程之上。

这组对照结果非常适合作为论文中的核心论点之一：

- warmup 负责“把球会往哪去学明白”
- RL 负责“把这套理解变成怎么接球的动作规律”

## 8. 因果遮挡实验

为了进一步回答“模型到底在用哪些内部量做动作决策”，本课题在真实 build 环境中进行了因果遮挡实验。与前面的 linear probe 不同，这一部分不是去“读取”表征中包含什么信息，而是直接在推理时对中间表示做干预，观察任务成功率如何变化。

### 8.1 实验设置

实验环境为：

- `ball_fetch/build/e2_360_0_6/ball_fetch.exe`

使用的策略来源为：

- `rl_be/results/exp_ec_0001/temp_realTime_0001/CarCatch/checkpoint.pt`

评估方式为冻结策略评估：

- 从训练好的 run `initialize-from`
- `learning_rate = 0`
- `beta = 0`
- `buffer_size > max_steps`

因此，在评估过程中不会发生参数更新，统计结果可以视为纯推理表现。

本次正式对比采用：

- `4` 个并行环境
- 每组 `5000` step

本轮最重要的是三种“只切读出、不破坏记忆更新”的遮挡：

1. `short_readout = 0`  
   `short` 仍然进入 `memory_fuse / GRU`，但动作头看不到 `short`。
2. `long_readout = 0`  
   `long` 仍然正常更新，但动作头看不到 `long`。
3. `self_readout = 0`  
   `self` 仍然参与记忆更新，但动作头看不到 `self`。

这样可以更精确地区分：

- 某一分支是否对动作读出本身关键
- 而不是把整条感知或记忆通路整体切断

### 8.2 结果

最终结果如下：

| condition | SuccessRate | Cumulative Reward | Episode Length |
|---|---:|---:|---:|
| baseline | 1.000 | 71.34 | 32.4 |
| short_readout = 0 | 0.533 | 5.88 | 33.6 |
| long_readout = 0 | 0.667 | 38.95 | 32.8 |
| self_readout = 0 | 1.000 | 70.86 | 33.2 |

### 8.3 结果解释

这组结果给出了比 linear probe 更强的因果证据：

1. `short -> action head` 是最关键的读出通路。  
   即使 `short` 仍然可以进入 `long memory`，只要动作头看不到 `short`，成功率仍然会从 `1.0` 大幅下降到约 `0.53`。

2. `long -> action head` 同样重要，但次于 `short`。  
   当动作头看不到 `long` 时，成功率下降到约 `0.67`，说明长时记忆确实在帮助动作决策，但作用略弱于最新短时视觉直连。

3. `self -> action head` 在当前环境里的贡献较小。  
   把 `self` 从动作读出中拿掉后，成功率基本保持 `1.0`，回报也只轻微下降。这与环境设定一致，因为当前动作本身就是速度指令，`self` 在这个任务里与动作控制高度相关，但并不构成决定性瓶颈。

### 8.4 对模型机制的启示

结合前面的 probe 与本轮因果遮挡，可以得到一个更完整的机制图景：

1. `short` 承载了当前球位置与落地时间等即时空间信息。
2. `long` 承载了轨迹趋势和未来判断。
3. 在真正输出动作时，模型更依赖 `short` 的即时直连，同时也显著依赖 `long` 的辅助判断。
4. `self` 更像一个辅助性的本体感觉通道，而不是这个任务中最核心的决策瓶颈。

因此，当前 agent 的接球策略并不是“只靠长记忆预判”，也不是“只靠当前一帧反应”，而是：

- 以最新短时视觉为主导
- 以长时记忆为重要补充
- 以自身状态为轻量辅助

这组因果遮挡结果，非常适合作为论文中“模型如何接球”的关键证据。

## 9. 泛化实验

为了评估该 agent 是否不仅能在训练环境中成功接球，还能在分布外环境中保持有效行为，本课题进一步在多个修改后的 Unity build 上进行了冻结策略评估。

### 9.1 实验设置

本轮泛化实验仍然使用：

- 原始训练好的完整模型
- `rl_be/results/exp_ec_0001/temp_realTime_0001/CarCatch/checkpoint.pt`
- `4` 个并行环境
- 每组 `5000` step
- 冻结策略评估，不进行任何参数更新

与前面的因果遮挡不同，这一轮**不使用任何遮挡**，只考察原始完整模型在分布偏移下的表现。

使用的测试环境包括：

- 原环境 `e2_360_0_6`
- `e2_360_0_6_cylinder`：抛射圆柱体
- `e2_360_0_6_cube`：抛射方块
- `e2_360_0_6_smallBall`：抛射更小的球
- `e2_360_0_6_fastg`：更大的重力加速度（2X）
- 原环境 + 输入灰度反转

其中需要说明的是，`smallBall` 在第一次评估时曾因 build 目录混淆而得到过一组旧结果。本文最终采用的是重新 build 后得到的重跑结果，旧结果不再使用。

### 9.2 结果

最终结果如下：

| environment | SuccessRate | Cumulative Reward | Episode Length |
|---|---:|---:|---:|
| baseline | 1.000 | 71.34 | 32.40 |
| cylinder | 0.500 | 17.08 | 34.07 |
| cube | 0.800 | 50.35 | 33.64 |
| smallBall | 0.533 | -1.18 | 33.87 |
| fastg | 0.947 | 64.87 | 23.89 |
| grayscale invert | 0.000 | -113.71 | 23.10 |

### 9.3 结果分析

从这组结果可以得到几个比较清楚的现象：

1. 模型对动力学偏移的鲁棒性明显强于对外观尺度偏移的鲁棒性。  
   在 `fastg` 环境中，尽管球下落速度更快，但成功率仍然达到约 `0.95`；相比之下，在 `smallBall` 环境中，成功率下降到约 `0.53`。

2. 对形状变化有一定泛化能力，但并不稳定。  
   `cube` 的表现尚可，成功率约 `0.80`；而 `cylinder` 下降到约 `0.50`。这说明模型并不是完全依赖特定球体外观，但对轮廓和视觉运动模式的改变仍然较敏感。

3. 对低层像素分布变化极其敏感。  
   灰度反转后，成功率直接降到 `0`，累计回报显著为负。这说明当前视觉编码器虽然学到了较强的任务表征，但仍然明显依赖训练时的亮度统计与对比度分布。

4. `fastg` 的 episode length 明显更短。  
   这与环境修改本身一致，因为更大的重力会让目标更快落地。也就是说，模型虽然在时间尺度改变后仍能接住大多数目标，但它可利用的决策窗口明显缩短了。

### 9.4 泛化结论

综合来看，当前 agent 的泛化能力具有明显的“选择性”：

- 对动力学变化有较强鲁棒性
- 对形状变化有中等鲁棒性
- 对目标尺度缩小较敏感
- 对像素反转这类光照/亮度分布变化几乎没有鲁棒性

这说明该模型已经不只是记住训练环境中的单一路径，但它的视觉表征仍然主要建立在训练分布附近，尤其缺乏对低层外观统计变化的稳健性。

从论文写作角度，这一结果也很有价值，因为它表明：

1. 模型确实学到了一部分超越单一环境的可迁移规律。
2. 这种可迁移性更偏向动力学与轨迹层面，而不是完全外观无关。
3. 后续若要进一步提升泛化性，最值得考虑的方向包括：视觉域随机化、亮度/对比度增强、目标外观多样化训练，以及更强的表征不变性约束。
