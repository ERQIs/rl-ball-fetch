# RL Ball Fetch（论文实验复现仓库）

## 研究问题

研究问题：

- 在移动接球任务中，什么样的视觉表示更有利于**数据效率**和**快速适配**？
- 相比直接端到端或显式状态输入，是否可以通过视觉-动作接口与空间先验得到更稳、更省样本的学习过程？

## 任务设定（与论文一致）

- 单目相机固定在小车上，篮筐刚性安装在车体上
- 球从远处抛来（文档中约 20m 飞行，约 2.2s）
- 控制输入是二维平面速度 `v_t ∈ R^2`（有速度上限）
- 目标是在接球事件时刻让球与篮筐对齐（终端误差最小化）

## 当前已复现的两组基线实验

1. 实验 1：显式状态输入基线（PPO + MLP）
   - 观测：12 维向量（车/球速度 + 相对位置）
   - 动作：2 维连续速度
   - 结果：成功率约 `74%`
   - 作用：作为后续方法对照组

2. 实验 2：纯视觉端到端基线（PPO + 简单 CNN）
   - 观测：灰度 `64x64`，帧堆叠 `K=6`
   - 编码器：2 层卷积 + FC(128) + Actor/Critic 头
   - 结果：成功率约 `74%`，训练曲线更稳定
   - 作用：验证纯视觉 RL 可行，为空间先验方法提供 baseline

## 本仓库在论文路线中的位置

当前仓库承担的是：

- 统一仿真环境与训练接口
- 复现并固化实验 1 / 实验 2
- 为后续“空间先验注入模型 + 表征分析”提供可对比基线

对应论文后续工作：

1. 奖励设计与训练策略系统优化
2. 空间先验注入的模型设计与实现
3. 视觉基线策略的中间表征分析与消融

## 仓库结构

- `ball_fetch/`：Unity 场景、Agent、采样与交互逻辑
- `rl_be/`：ML-Agents 训练配置、脚本、训练说明
- `vis_backbone/`：空间/时序一致性视觉表征（Stage A）设计文档
- `docs/`：从 prompt 整理后的实验说明文档

## 文档导航

- 训练与环境细节：`rl_be/README.md`
- 显式状态基线（Tracking v0）：`docs/README_tracking_v0.md`
- 纯视觉训练流水线：`docs/README_vision_pipeline.md`
- Stage A 视觉表征学习：`vis_backbone/README.md`

## 快速开始

先跑通对照基线：

1. 打开 Unity `ball_fetch` 项目并进入 `CatchCarScene`
2. 按 `rl_be/README.md` 启动 PPO 训练
3. 先复现实验 1（状态输入），再切到实验 2（视觉输入）




总问题：
在移动接球任务中，什么样的视觉表征/视觉接口
能够支持 data-efficient learning 与 few-shot adaptation？

│
├── Part I. 空间表征：先把“看见什么”这件事做好
│   │
│   ├── 问题
│   │   单帧视觉输入下，什么样的视觉表征更适合作为控制接口？
│   │
│   ├── 核心想法
│   │   在视觉头中引入空间连续性先验（spatial continuity prior），
│   │   保留 feature map 的空间结构，而不是过早压成全局向量。
│   │
│   ├── 方法
│   │   self-supervised backbone pretraining
│   │   - grayscale input: (1, 64, 64)
│   │   - feature map: (8, 8, 8)
│   │   - reconstruction
│   │   - feature transport consistency
│   │   - warp-decoding consistency
│   │
│   ├── 结论
│   │   融合空间连续性先验的视觉表征
│   │   能提升移动接球任务中的 sample efficiency。
│   │
│   └── 意义
│       说明“结构化的空间视觉接口”是有效的第一步。
│
├── Part II. 时序表征：进一步解决“运动如何被理解”
│   │
│   ├── 动机
│   │   接球本质上是运动理解问题；
│   │   单帧只能看到位置，难以直接推断速度、方向、轨迹趋势。
│   │
│   ├── 为了降低复杂度，先研究一个简化 setting
│   │   让小车先静止观察一段时间（例如 1s），
│   │   再输出一个速度并执行固定时长。
│   │
│   ├── 这样做的好处
│   │   - 去掉 ego-motion 对视觉的干扰
│   │   - 让“从时序中提取 motion cue”成为主问题
│   │   - 更适合做 clean comparison 和 few-shot adaptation
│   │
│   ├── 研究问题
│   │   什么样的 temporal visual representation
│   │   最适合从静态观察序列中提取运动信息？
│   │
│   ├── 对比路线
│   │   - single frame
│   │   - frame stack
│   │   - RNN / GRU
│   │   - 其他时序架构（老板推荐的模型）
│   │   - 普通 encoder vs continuity encoder 作为 temporal input
│   │
│   └── 目标
│       不只是证明“多帧比单帧好”，
│       而是找出“哪种时序表征最有利于 data-efficient control”。
│
├── Part III. 训练策略：把学习资源聚焦到“没学会”的地方
│   │
│   ├── 动机
│   │   哪里 loss 高 / error 大，说明模型还没学会；
│   │   应该把更多训练资源放在那里。
│   │
│   ├── 可能做法
│   │   - prioritized replay
│   │   - failed episode oversampling
│   │   - large terminal-miss case prioritization
│   │   - high TD-error transition prioritization
│   │
│   ├── 作用
│   │   不改变 reward 定义，
│   │   只改变训练关注点，提高学习效率。
│   │
│   └── 目标
│       研究 focused training 是否能进一步提升
│       temporal/spatial representation 的下游效果。
│
└── Part IV. Few-shot adaptation：回到最终目标
    │
    ├── 上层目标
    │   当小车动力学、速度参数、摩擦等发生变化时，
    │   能否只用少量交互快速完成适配？
    │
    ├── 与前面工作的关系
    │   - Part I 提供 calibration-friendly 的空间视觉接口
    │   - Part II 提供 motion-aware 的时序表征
    │   - Part III 提供更高效的训练策略
    │
    ├── 最终问题
    │   什么样的视觉接口最适合 few-shot calibration/adaptation？
    │
    └── 毕设总故事落点
        从“空间连续性先验”出发，
        逐步走向“运动表征”与“少样本适配”，
        最终服务于 embodied visual-motor adaptation。