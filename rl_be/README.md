# RL Ball Fetch 实验说明（Unity + ML-Agents）

## 1. 实验目的

本项目目标是训练一个小车智能体，在 Unity 物理环境中移动并接住抛出的球。  
核心任务可定义为：

- 输入：环境状态（当前为向量观测）
- 输出：左右轮离散动作
- 目标：最大化累计奖励（尽可能多接球、尽可能少漏接）

---

## 2. 实验环境

### Unity 侧

- Unity Editor: `6000.3.10f1`
- Unity Package: `com.unity.ml-agents = 4.0.2`
- 主训练行为名（Behavior Name）: `CarCatch`
- 场景：`CatchCarScene`

### Python 侧

- 推荐 Python 版本：`3.10.x`
- 关键依赖（见 `requirements.txt`）：
  - `mlagents==1.1.0`
  - `torch==2.1.2`
  - `numpy==1.23.5`
  - `tensorboard>=2.12,<3`
  - `setuptools<81`

说明：该依赖组合用于避免 checkpoint/ONNX 导出阶段的兼容性问题。

---

## 3. 小车与环境配置（Unity）

### 3.1 CarCatcherAgent（小车）

主要参数（当前场景序列化值）：

- `wheelForce = 25`
- `maxSpeed = 8`
- `angularDamping = 0.2`
- `arenaRadius = 6`
- `centerOfMassLocal = (0, -0.25, 0)`

动作执行频率：

- `DecisionPeriod = 1`（每个物理步都做决策）

### 3.2 球生成与轨迹（BallSpawner / Ball）

BallSpawner 参数（当前场景序列化值）：

- `spawnHeight = 2`
- `minForwardDist = 6`
- `maxForwardDist = 10`
- `sideJitter = 2`
- `Tmin = 2.5`
- `Tmax = 3`
- `landingNoise = 3`

Ball 逻辑：

- 根据目标落点与飞行时间反推初速度，使球以抛体轨迹飞向篮筐附近
- 球触地（Ground）判定为 miss，触发失败奖励与回合结束

---

## 4. 模型输入输出

## 4.1 输入（Observation）

当前不是图像输入，而是 **12 维向量输入**（`CollectObservations`）：

1. `relPos`：球相对篮筐中心位置（3 维）
2. `relVel`：球相对小车速度（3 维）
3. `transform.forward`：小车朝向（3 维）
4. `rb.linearVelocity`：小车线速度（3 维）

合计：`12` 维。

## 4.2 输出（Action）

动作为离散动作，配置为：

- 连续动作数：`0`
- 离散分支：`[5, 5]`（左右轮各 5 档）

代码映射：

- `DiscreteActions[0]` -> 左轮
- `DiscreteActions[1]` -> 右轮
- 档位 `0..4` 映射到控制量 `[-1, -0.5, 0, 0.5, 1]`

---

## 5. 奖励函数

当前奖励由三部分组成：

1. 每步惩罚（step penalty）  
   - `r_step = stepPenalty = -0.001`

2. 距离惩罚（稠密项）  
   - `r_dist = -distance(ball, basketCenter) * distanceRewardScale`
   - `distanceRewardScale = 0.002`

3. 终局奖励  
   - 接到球：`catchReward = +3.0`
   - 漏接球：`missReward = -1.0`

因此可写作：

- 非终局步：`r_t = -0.001 - 0.002 * d_t`
- 接球终局额外加：`+3.0`
- 漏接终局额外加：`-1.0`

---

## 6. 训练模型配置（PPO）

配置文件：`config/catch_ppo.yaml`

- `trainer_type: ppo`
- `batch_size: 512`
- `buffer_size: 10240`
- `learning_rate: 3e-4`
- `beta: 5e-3`
- `epsilon: 0.2`
- `lambd: 0.95`
- `num_epoch: 3`
- `learning_rate_schedule: linear`
- `normalize: true`
- `hidden_units: 64`
- `num_layers: 8`
- `gamma: 0.99`
- `checkpoint_interval: 50000`
- `max_steps: 500000`
- `time_horizon: 128`
- `summary_freq: 5000`

引擎设置：

- `time_scale: 20`
- `target_frame_rate: -1`
- `capture_frame_rate: 60`

---

## 7. 如何开始训练

## 7.1 在 Unity Editor 中训练

1. 打开 `CatchCarScene`
2. 点击 Unity `Play`
3. 在 `rl_be` 目录启动训练：

```powershell
.\.venv\Scripts\python.exe -m mlagents.trainers.learn .\config\catch_ppo.yaml --run-id carcatch_v2 --force --timeout-wait 300
```

继续训练（resume）：

```powershell
.\.venv\Scripts\python.exe -m mlagents.trainers.learn .\config\catch_ppo.yaml --run-id carcatch_v2 --resume --timeout-wait 300
```

## 7.2 使用脚本训练（默认 .venv310）

仓库内脚本默认使用 `.venv310`：

- `scripts/setup_env.ps1`
- `scripts/train_editor.ps1`
- `scripts/train_build.ps1`
- `scripts/tensorboard.ps1`

如果你当前使用的是 `.venv`，建议直接使用上面的 Python 命令，或将脚本中的 `.venv310` 改为 `.venv`。

---

## 8. 日志与结果

- 训练输出目录：`rl_be/results/<run-id>/`
- TensorBoard：

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir .\results --port 6006
```

浏览器打开：`http://localhost:6006`

