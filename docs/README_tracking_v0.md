# Tracking v0 方案说明

## 1. 任务定义

- 机器人形态：`mobile base + 固定接球点（篮筐中心）`
- 成功条件：球进入接球触发区（或触发 catch collider）
- 失败条件：球触地、超时、出界

这个方案等价于把 Catch It 的 Stage-1 思路迁移到当前项目：原任务里的末端触碰目标，映射为本项目中的“接球点对齐并接住球”。

## 2. 观测设计（状态输入）

推荐使用以下最小可跑通观测（约 12~14 维）：

- `delta_pos = p_ball - p_catchPoint`
- `ball_vel`
- `base_lin_vel`
- `base_yaw_sin_cos = [sin(yaw), cos(yaw)]`
- `delta_dir_in_base = normalize(delta_pos)`（可选）

设计原则：

- 尽量用 base 坐标系，提升稳定性
- 用 `sin/cos` 表示朝向，减少角度不连续问题

## 3. 动作设计

使用连续动作速度控制：

- `a = [v_forward, v_lateral]`

建议放弃差速轮式离散控制，直接输出平面速度，收敛更稳定，也更接近后续真实控制接口。

## 4. 策略网络

PPO + MLP（Actor/Critic 分离）：

- Actor: `[256, 128]` -> 动作均值
- Critic: `[256, 128]` -> 值函数
- Observation normalization: 开启
- Action std: 可学习或设较大初值

## 5. 奖励函数

定义：

- `d = ||p_ball - p_catchPoint||`
- `r_pos = d_prev - d`（奖励“进步”）
- `r_prec = exp(-k*d)`（近距离精度 shaping）
- `r_ctrl = -c*||a||^2`（动作惩罚）

终局奖励：

- success: `+5`（可先加大，优先跑通）
- fail: `-2 ~ -5`

一个常用组合：

- `R = 10*r_pos + 2*r_prec + r_ctrl + R_terminal`

## 6. PPO 推荐超参

- `gamma=0.99`
- `lambda=0.95`
- `lr=3e-4 ~ 5e-4`
- `time_horizon=64~128`
- `batch_size=512`
- `buffer_size>=10240`
- `num_epoch=3~6`
- `entropy(beta)=0.001~0.005`

## 7. 跑通优先的关键点

1. 连续速度动作替代差速轮控制
2. 奖励从“纯距离惩罚”改为“距离进步 + 精度 shaping”
3. 成功奖励先给足，再做细调

