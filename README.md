# RL Ball Fetch

一个基于 Unity ML-Agents 的接球强化学习项目。目标是训练小车在球飞行阶段移动到合适位置，并完成接球。

## 项目目标

- 任务：移动小车接住抛来的球（球进入接球触发区视为成功）。
- 当前基线：状态输入（12 维）+ PPO，已可稳定训练。
- 当前方向：从状态控制扩展到纯视觉控制（CNN + PPO / BC）。

## 方法总览

项目分为两条主线：

1. Tracking v0（状态输入）
   - 输入：球相对位置/速度 + 车体速度等低维状态。
   - 输出：连续速度动作（前后 + 左右）。
   - 训练：PPO，使用“距离进步 + 精度 shaping + 控制惩罚 + 终局奖励”的奖励设计。

2. Vision Pipeline（纯视觉）
   - 输入：90x90 图像 + 帧堆叠（默认 K=6, stride=2）。
   - 模型：CNN backbone + Actor/Critic MLP。
   - 训练路径：PPO from scratch、BC、BC 预训练后 PPO 微调。

## 仓库结构

- `ball_fetch/`：Unity 场景与 Agent 逻辑。
- `rl_be/`：ML-Agents 训练配置与脚本。
- `vis_backbone/`：视觉表征学习（Stage A）设计文档。
- `docs/`：面向项目成员的人类可读说明文档。

## 文档入口

- Tracking v0 设计说明：`docs/README_tracking_v0.md`
- 纯视觉训练流水线：`docs/README_vision_pipeline.md`
- Stage A 视觉表征学习：`vis_backbone/README.md`
- 训练与环境操作细节：`rl_be/README.md`

## 你应该先看哪份

- 想先把环境跑通：先看 `rl_be/README.md` + `docs/README_tracking_v0.md`
- 想做纯视觉训练：看 `docs/README_vision_pipeline.md`
- 想做视觉表征预训练：看 `vis_backbone/README.md`

