# RL Ball Fetch 实验说明（Tracking v0）

## 1. 实验目标

训练一个小车智能体在球飞行过程中移动到底盘上的接球点（basket center）附近，并最终让球进入接球触发区。

- 成功：球进入 `CatchZone`（触发 `OnBallCaught`）
- 失败：球触地（`OnBallMissed`）/ 小车出界 / 回合超时

---

## 2. 环境与依赖

### Unity

- Unity Editor: `6000.3.10f1`
- ML-Agents Package: `com.unity.ml-agents 4.0.2`
- Behavior Name: `CarCatch`
- Scene: `CatchCarScene`

### Python

- Python: `3.10.x`
- 关键依赖（`requirements.txt`）：
  - `mlagents==1.1.0`
  - `torch==2.1.2`
  - `numpy==1.23.5`
  - `tensorboard>=2.12,<3`
  - `setuptools<81`

---

## 3. Agent 与场景配置

## 3.1 小车控制（连续动作）

`CarCatcherAgent` 采用连续速度控制，不再使用差速轮离散动作：

- `maxForwardSpeed = 8`
- `maxLateralSpeed = 8`
- `keepFixedHeading = true`（固定朝向，不允许旋转）
- `arenaRadius = 6`
- `defaultMaxStep = 600`

手动模式（Heuristic）：

- `W/S`：前后移动
- `A/D`：左右平移

## 3.2 球生成

`BallSpawner` 在小车前方随机位置抛球，并给定随机飞行时间：

- `spawnHeight = 2`
- `minForwardDist = 6`
- `maxForwardDist = 10`
- `sideJitter = 2`
- `Tmin = 2.5`
- `Tmax = 3`
- `landingNoise = 3`

---

## 4. 模型输入输出

## 4.1 Observation（12 维）

每步观测：

1. `delta_pos = p_ball - p_catchPoint`（3）
2. `ball_vel`（3）
3. `base_vel`（3）
4. `dir_to_ball = normalize(delta_pos)`（3）

总计：`12` 维向量。

## 4.2 Action（连续 2 维）

`a = [v_forward, v_lateral]`，每维 `[-1, 1]`，映射到目标平面速度：

- `v_forward * maxForwardSpeed`
- `v_lateral * maxLateralSpeed`

---

## 5. 奖励函数（Tracking 风格）

在 `OnActionReceived` 中使用：

- `d = ||p_ball - p_catchPoint||`
- `r_pos = d_prev - d`（鼓励“进步”）
- `r_prec = exp(-k*d)`（近距离精度 shaping，`k=precisionK`）
- `r_ctrl = -c*||a||^2`（控制惩罚，`c=controlPenaltyScale`）

每步奖励：

- `r_step = progressRewardScale * r_pos + precisionRewardScale * r_prec + r_ctrl`

终局奖励：

- success：`catchReward = +5`
- miss：`missReward = -3`
- out of arena：`outOfArenaPenalty = -3`

默认权重：

- `precisionK = 3`
- `progressRewardScale = 10`
- `precisionRewardScale = 2`
- `controlPenaltyScale = 0.02`

---

## 6. PPO 配置（`config/catch_ppo.yaml`）

- `trainer_type: ppo`
- `batch_size: 512`
- `buffer_size: 10240`
- `learning_rate: 3e-4`
- `beta: 5e-3`
- `epsilon: 0.2`
- `lambd: 0.95`
- `num_epoch: 3`
- `normalize: true`
- `hidden_units: 256`
- `num_layers: 2`
- `gamma: 0.99`
- `time_horizon: 128`
- `max_steps: 500000`
- `checkpoint_interval: 50000`
- `summary_freq: 5000`

引擎参数：

- `time_scale: 20`
- `target_frame_rate: -1`
- `capture_frame_rate: 60`

---

## 7. 如何开始训练

## 7.1 Editor 训练（推荐）

1. 打开 Unity 项目 `ball_fetch`
2. 打开 `CatchCarScene`
3. 点击 Unity `Play`
4. 在 `rl_be` 目录执行：

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\.venv\Scripts\python.exe -m mlagents.trainers.learn .\config\catch_ppo.yaml --run-id carcatch_tracking_v1 --force --timeout-wait 300
```

继续训练：

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\.venv\Scripts\python.exe -m mlagents.trainers.learn .\config\catch_ppo.yaml --run-id carcatch_tracking_v1 --resume --timeout-wait 300
```

## 7.2 查看训练曲线

```powershell
cd d:\projects\rl-ball-fetch\rl_be
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\results --port 6006
```

打开：`http://localhost:6006`

---

## 8. 结果目录

- 训练输出：`rl_be/results/<run-id>/`
- 模型与 checkpoint：`rl_be/results/<run-id>/`

