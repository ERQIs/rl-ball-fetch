# 毕业论文最终实验矩阵（修补版）

## 1. 一句话主线

本文不是要构建一个面向接球应用的专用智能体，而是借助移动接球这一结构简洁、空间关系清晰、实验条件可控的任务设定，研究：

1. 空间连续性先验能否改善视觉表征质量；
2. 这种空间表征能否进一步支持更好的时序建模与动态趋势推断；
3. 上述表征优势是否能够体现为更高的数据效率。

## 2. 论文不再承担的目标

下面这些内容不应继续作为主线叙事：

- 不以 few-shot adaptation 作为当前论文的主问题
- 不与 RT-1、RT-2、OpenVLA、pi0 等大规模 VLA 模型做直接实验对比
- 不追求“把所有主流视觉头/时序模块都跑一遍”
- 不把所有 recipe tuning 都写成主结果

这些内容最多作为研究背景、动机或附录说明，不应继续扩成新的主实验坑。

## 3. 论文核心假设

### H1 空间表征假设

相比将图像快速压缩为全局向量的视觉接口，显式保留空间结构的视觉表征更适合作为具身控制的输入接口。

### H2 空间连续性假设

在保留空间结构的前提下，引入 continuity prior 可以进一步增强视觉表征的空间一致性与可迁移性。

### H3 时序延伸假设

建立在第一部分空间表征基础上的时序模型，能够在时间维度上形成更强的动态信息整合能力和目标趋势推断能力，并在小数据条件下体现出更高的数据效率。

## 4. 最终实验矩阵

| 编号 | 实验名称 | 主要回答的问题 | 对比组 | 关键指标 | 当前状态 | 论文角色 |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | 闭环控制动机基线 | 纯视觉控制是否可行，空间/视觉问题是否值得研究 | 状态输入 PPO vs 纯视觉 PPO | 成功率、训练稳定性 | 已有 | 动机实验 |
| E1 | 空间视觉头对比 | 空间结构是否重要，continuity prior 是否有效 | `Global-CNN` vs `Spatial-Map` vs `Continuity-Head` | 下游误差、数据效率、多种子稳定性 | `Spatial-Map` 对照缺失，建议补 | 第一部分主实验 |
| E2 | 时序必要性对比 | 单帧是否不足，时间信息是否必要 | `Single-Frame` vs `CNN+GRU` | `mean_l2`、`success@0.2` | 已有 | 第二部分入口实验 |
| E3 | 结构化时序模型基线 | 结构化时空状态是否优于普通时序基线 | `CNN+GRU` vs `Multiscale-Scratch` | `mean_l2`、`success@0.2`、参数量对比 | 基本已有 | 第二部分主对照 |
| E4 | 空间表征到时序建模的延伸 | 第一部分空间表征是否能支持更好的时序推断 | `Multiscale-Scratch` vs `Spatial-Initialized Temporal Model` | 20/40/100% 数据效率、测试性能 | 部分已有，需收束为单一主线 | 第二部分核心实验 |
| E5 | 迁移方式消融 | 预训练表征应作为 frozen feature 还是 finetuned init | `Frozen` vs `Finetune` | 同上 | 已有 | 关键消融 |
| E6 | 工程型后续 recipe | 不同预训练耦合方式谁更强 | `Stacked888` vs `Adapter` 等 | 同上 | 已有但较杂 | 附录/补充 |

## 5. 每组实验该怎么讲

### E0 闭环控制动机基线

用途不是证明“我做出了接球 agent”，而是说明：

- 纯视觉控制在这个任务上是可行的；
- 但视觉接口如何设计会显著影响学习过程；
- 因此有必要进一步研究视觉表征本身。

推荐引用材料：

- [README.md](D:/projects/rl-ball-fetch/README.md)
- [rl_be/README.md](D:/projects/rl-ball-fetch/rl_be/README.md)

### E1 空间视觉头对比

这是现在最应该补的实验。

如果你想证明“空间连续性先验有效”，至少需要三组：

1. `Global-CNN`
说明传统做法：图像经过 CNN 后做全局池化或展平，输出向量。

2. `Spatial-Map`
保留 `8x8` 或类似 feature map，但不加 continuity prior。
这组是最关键的控制组，用来区分：
到底收益来自“保留空间结构”，还是来自“continuity prior”。

3. `Continuity-Head`
你的完整第一部分方法。

如果时间非常紧，这一组实验甚至比继续补新的时序模型更重要。

### E2 时序必要性对比

这组实验已经能回答一个非常关键的问题：

- 单帧视觉不足以稳定完成该任务；
- 时间信息是必要的；
- 因而第二部分不是“多做一点模型”，而是在回答一个确实存在的问题。

推荐引用材料：

- [observe_then_act_experiment_report.md](D:/projects/rl-ball-fetch/temperal/experiments/observe_then_act/reports/observe_then_act_experiment_report.md)

### E3 结构化时序模型基线

这一组的目标不是证明你比“所有时序模型”都强，而是证明：

- 普通时序聚合基线已经很强，不能拿弱 baseline 垫；
- 在此基础上，结构化时空状态仍然有研究价值；
- 你的模型优势来自结构偏置，而不只是更大参数量。

这里 `CNN+GRU` 已经是一个足够合理、足够主流、也足够好写的 baseline。
不建议为了“显得全面”再硬补一堆 Transformer、Mamba 或 TimeSformer。

### E4 空间表征到时序建模的延伸

这是第二部分最重要的主实验，最好写成：

“建立在第一部分空间表征基础上的时序模型，是否能够获得更好的动态趋势推断能力和数据效率？”

推荐只保留一条最干净的主线，不要把所有 pretraining route 都摆成并列主方法。

当前更适合作为主线的候选有两种：

1. `Stacked888`
优点：故事最顺，最符合“Part II 建立在 Part I 基础之上”的叙事。
适合写成主线方法。

2. `Adapter`
优点：当前整体结果更强。
缺点：工程味更重，和 Part I 的直接承接关系稍弱。

推荐写法：

- 主文主线优先采用 `Stacked888` 或最贴近“空间先验 -> 时序建模”的版本；
- `Adapter` 放在附录或补充实验里，作为“更强但更工程化的 recipe”。

### E5 迁移方式消融

这组已经很有价值，因为它告诉你：

- 你的表征目前还不是强到可以直接 frozen 的通用特征；
- 但它作为可微调初始化是有明显价值的。

这正好能支撑一个很真实、也很学术的结论：

“该表征更适合作为时序建模的初始化先验，而非开箱即用的固定特征抽取器。”

### E6 工程型 recipe 对比

这一组不要再当主线扩写。

建议处理方式：

- 主文只保留 1 个最好解释的代表方法
- 其他 recipe 放到附录
- 在正文中只给一句总结：
  “我们进一步尝试了若干预训练耦合方案，结果见附录；整体趋势支持主文结论，但不同 recipe 的收益具有明显实现依赖性”

## 6. 现在最值得补的 3 个实验

如果你现在时间有限，我建议只补下面三项。

### 必补 1：空间头控制实验

`Global-CNN` vs `Spatial-Map` vs `Continuity-Head`

这是第一部分是否成立的关键证据。

### 必补 2：同一时序模块下的输入接口对比

固定同一个 temporal model，只替换输入视觉表征：

- 普通视觉输入
- 空间连续性视觉输入

这样可以直接回答：

“第一部分学到的空间表征，是否真的帮助了第二部分的时序推断”

### 必补 3：多种子数据效率曲线

至少对最关键的主比较，补：

- `20%`
- `40%`
- `100%`
- `3 seeds`

你这篇论文很适合打“数据效率”这张牌，所以多种子和数据效率曲线比再补一个新模型更值钱。

## 7. 可以直接作为论文主表的内容

### 主表 A：空间视觉头比较

- 模型
- 是否保留空间结构
- 是否使用 continuity prior
- 下游误差
- 小数据性能

### 主表 B：时序模型比较

- `Single-Frame`
- `CNN+GRU`
- `Multiscale-Scratch`
- `Multiscale + Prior Init`

### 主表 C：迁移方式消融

- `Scratch`
- `Frozen`
- `Finetune`

### 主图 D：数据效率曲线

- 横轴：训练数据比例或样本数
- 纵轴：`mean_l2` / `success@0.2`

## 8. 建议采用的最终叙事结构

### 第一部分

先回答：

“什么样的视觉接口更适合具身控制中的空间能力学习？”

### 第二部分

再回答：

“建立在第一部分空间表征基础上的时序模型，能否形成更好的动态信息整合与趋势推断能力？”

### 第三部分（实验总结，而不是新方法）

最后回答：

“这些表征优势是否体现为更高的数据效率？”

这样你整篇论文就从“无头苍蝇式堆实验”变成了一个逐层推进的研究链条：

- 先做空间表征
- 再做时序延伸
- 最后看数据效率

## 9. 现有材料如何归位

### 可以直接保留为主文证据

- [README.md](D:/projects/rl-ball-fetch/README.md)
- [rl_be/README.md](D:/projects/rl-ball-fetch/rl_be/README.md)
- [observe_then_act_experiment_report.md](D:/projects/rl-ball-fetch/temperal/experiments/observe_then_act/reports/observe_then_act_experiment_report.md)
- [formal_results.md](D:/projects/rl-ball-fetch/temperal/experiments/observe_then_act/reports/formal_results.md)

### 更适合作为附录或补充材料

- [continuity_plus_multiscale_progress.md](D:/projects/rl-ball-fetch/temperal/experiments/continuity_prior/notes/continuity_plus_multiscale_progress.md)

原因：

- 它很有价值
- 但更像探索过程记录，而不是主文最简洁的证据链

## 10. 一句话结论

你现在不需要“补齐所有主流方法”，你需要的是：

**用最少但最关键的控制实验，把“空间连续性先验 -> 更好的时序建模/数据效率”这条因果链补完整。**
