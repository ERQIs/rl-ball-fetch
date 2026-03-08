# 纯视觉训练流水线说明

## 1. 目标

在 Unity 接球环境中实现完整的纯视觉训练流程：

- 采样
- 模型构建
- 训练（PPO / BC / BC->PPO）
- 评估
- 日志与 checkpoint

## 2. 环境约束（当前版本）

- 渲染帧率：约 60 FPS
- 动作更新：50 Hz（支持 action repeat / decision period 调整）
- 动作形式：二维速度向量
- 图像尺寸：默认 `90x90`

## 3. 观测与预处理

默认输入：

- RGB 图像（`90x90`，归一化到 `[0,1]`）
- 帧堆叠：`K=6`，`stride=2`
- 输出形状：`(3*K, 90, 90) = (18, 90, 90)`

可选输入：

- 追加最小 proprioception（车体速度 2~3 维），与视觉 embedding 拼接

## 4. 模型结构

### CNN backbone（示例）

- `Conv(32, 8x8, stride=4) -> ReLU`
- `Conv(64, 4x4, stride=2) -> ReLU`
- `Conv(64, 3x3, stride=1) -> ReLU`
- `Flatten -> FC(256) -> ReLU`

### Actor-Critic Head

- Actor: `FC(256)->ReLU->FC(128)->ReLU->action_mean + learnable log_std`
- Critic: 类似结构输出 `V(s)`

## 5. 三种训练模式

1. PPO from scratch（纯视觉 RL 基线）
2. BC（用专家策略或人工示范数据做监督学习）
3. BC 预训练 + PPO 微调（推荐）

## 6. 推荐 PPO 超参（起点）

- `learning_rate=3e-4`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `entropy_coef=0.005`
- `value_coef=0.5`
- `max_grad_norm=0.5`
- `rollout_length=2048`
- `minibatch=64/128`
- `epochs=10`

## 7. BC 数据与训练要点

数据来源优先级：

1. 已训练 state policy 作为 teacher 自动采样
2. 键盘/手柄人工示范

数据字段至少包含：

- `stacked_obs`
- `action`
- `episode_id`
- `timestep`

训练配置建议：

- Loss: `MSE(action_pred, action_expert)`
- Optimizer: Adam
- `lr=1e-3`
- `batch=256`

## 8. 评估与日志

固定间隔评估，输出：

- `success_rate`
- `mean_episode_reward`
- `mean_episode_length`

日志系统：

- TensorBoard 或 W&B（二选一，配置可开关）
- 保存最佳 success_rate checkpoint 和最后 checkpoint

## 9. 推荐命令示例

- `python train_ppo.py --config configs/ppo_vision.yaml`
- `python train_bc.py --config configs/bc_vision.yaml`
- `python train_ppo.py --config configs/ppo_vision_from_bc.yaml --init_ckpt <bc_ckpt>`

## 10. 成功判据

- 纯视觉 PPO 出现非零成功率且持续上升
- BC 能复现部分追球/接球行为
- BC->PPO 相比纯 PPO 有更好样本效率

