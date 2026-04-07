# Embodied Cognition Analysis Plan

## 核心问题

针对当前接球 agent，最值得回答的不是“它会不会接球”，而是以下三个问题：

1. 它的内部表征里是否编码了空间量？
2. 这些空间量主要存在于短时视觉表征，还是长时记忆表征？
3. 动作到底更依赖最新视觉，还是更依赖累积记忆？

## 分析拆解

### 一类问题：表征中有没有空间信息

可检验目标：

- 落点相对位置 `landing_offset`
- 球相对车的位置 `ball_rel_pos`
- 球相对速度 `ball_rel_vel`
- 自身速度 `self_velocity`
- 剩余落地时间 `time_to_land`

方法：

- 从模型内部提取 `z_short`、`z_self`、`z_mem`、`h_long`
- 用线性 probe 检查这些表征对上述目标的可解码程度

如果某个变量可以从某层表征中被高质量线性恢复，说明这一层已经显式编码了相关空间结构。

### 二类问题：短时模块和长时模块分别负责什么

重点比较：

- `z_short`
- `h_long`
- `readout = [z_short, h_long, z_self]`

关注现象：

- `z_short` 是否更擅长编码当前球的位置
- `h_long` 是否更擅长编码落点或剩余时间
- `z_self` 是否主要承载本体感觉，而不是外界空间信息

### 三类问题：动作是怎么被决定的

重点做因果式分析：

- 将 `z_short` 置零，看动作变化
- 将 `h_long` 置零，看动作变化
- 将 `z_self` 置零，看动作变化

这能帮助判断：

- 动作是否主要依赖最新视觉修正
- 动作是否主要依赖长时轨迹记忆
- 自身状态在控制中的必要性有多强

## 已完成的第一步

已新增脚本：

- `temperal/experiments/embodied_control/scripts/analyze_spatial_probes.py`

该脚本可以：

1. 从离线轨迹中重放模型
2. 提取 `short/self/memory/long/readout` 表征
3. 拟合线性 probe
4. 输出各类空间量的测试集 `R^2`

## 初步发现

基于小样本 smoke 测试，已经观察到：

- `short` 对 `ball_rel_pos` 的解码能力很强
- `short` 对 `time_to_land` 也有较强信息
- `self` 几乎只编码 `self_velocity`
- `long` 对 `landing_offset`、`ball_rel_vel`、`time_to_land` 比 `short` 更有优势

这初步支持如下解释：

1. 短时视觉表征更像“当前看见的球在哪里”
2. 长时记忆更像“对轨迹趋势和未来结果的累积判断”
3. 自身状态分支主要承担本体感觉，而不是替代视觉空间建模

## 下一步建议

1. 用完整数据集重新跑 probe，而不是只用 smoke 子集。
2. 增加动作因果分析脚本，测量 `z_short / h_long / z_self` 对动作输出的影响。
3. 做遮挡实验，观察模型是否依赖球在图像中的某些固定区域。
4. 做时序打乱实验，检验模型是否真的利用了运动连续性。
5. 将结果整理成论文中的“表征分析”章节。
