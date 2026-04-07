# Smoke Experiment Results

## 1) 代码路径

- Dataset loader: `temperal/src/datasets/trajectory_dataset.py`
- 单帧模型: `temperal/src/models/single_frame_regressor.py`
- CNN+GRU 模型: `temperal/src/models/cnn_gru_regressor.py`
- 训练入口: `temperal/scripts/train.py`
- 评估入口: `temperal/scripts/evaluate.py`
- Split 生成脚本: `temperal/scripts/make_splits.py`
- Smoke test loader: `temperal/scripts/test_data_loader.py`

## 2) Smoke 实验配置

### single_frame_smoke.yaml
- dataset.root: `D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260314_140730`
- observation_length: 8
- frame_stride: 2
- use_last_n_frames: true
- img_size: 64
- model.name: `single_frame`
- epochs: 1
- batch_size: 16
- lr: 1e-3
- output_dir: `temperal/outputs/single_frame_smoke`

### gru.yaml（已在实验中运行）
- model.name: `cnn_gru`
- observation_length: 16
- frame_stride: 1
- epochs: 10
- batch_size: 8
- lr: 5e-4
- output_dir: `temperal/outputs/gru_exp`

## 3) Smoke实验命令

- 训练单帧 smoke：
  ```bash
  python temperal/scripts/train.py --config temperal/configs/single_frame_smoke.yaml
  ```
- 评估单帧 smoke：
  ```bash
  python temperal/scripts/evaluate.py --config temperal/configs/single_frame_smoke.yaml --checkpoint temperal/outputs/single_frame_smoke/last.pt
  ```
- 训练 GRU baseline：
  ```bash
  python temperal/scripts/train.py --config temperal/configs/gru.yaml
  ```
- 评估 GRU baseline：
  ```bash
  python temperal/scripts/evaluate.py --config temperal/configs/gru.yaml --checkpoint temperal/outputs/gru_exp/last.pt
  ```

## 4) 关键结果

### Single-frame baseline
- mean_l2: ~0.99
- median_l2: ~1.00
- success_at_0.2: ~0.09

### CNN+GRU baseline
- mean_l2: 0.0293
- median_l2: 0.02685
- success_at_0.2: 1.0

## 5) 结论与下一步

- 结果显示：当前 CNN+GRU 模型在这个数据上比 single-frame 模型显著好。
- 下一步建议：用 `temperal/configs/gru.yaml` 做多个 observation_length/frame_stride 组合的网格实验，记录指标。
- 结果保存在：`temperal/outputs/<exp>/last.pt` 以及评估打印结果。

---

## 6) 2026-03-16 更新：前 2/3 轨迹观测 + train/val split

这轮 smoke test 对任务设定做了两处关键调整：

- 观测窗口不再取轨迹尾段，而是只允许看到轨迹前 `67%` 的可见部分。
- 在这个可见前缀中均匀采样固定长度序列，而不是取最后 `N` 帧。
- 训练和评估开始真正使用 split：
  - train: `temperal/data/splits/train.txt`（350 条）
  - val: `temperal/data/splits/val.txt`（75 条）
  - test: `temperal/data/splits/test.txt`（75 条）

### 更新后的配置

#### single_frame_smoke.yaml
- observation_length: 8
- frame_stride: 2
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 1
- batch_size: 16
- lr: 1e-3

#### gru.yaml
- observation_length: 16
- frame_stride: 1
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 10
- batch_size: 8
- lr: 5e-4

### 抽样检查

- loader 抽样示例：`[0, 5, 10, 15, 21, 26, 31, 36, 41, 46, 51, 56, 62, 67, 72, 77]`
- 原始轨迹长度约 `115-117` 帧，因此当前设定相当于“观察前约 2/3，再行动”。

### 这轮 smoke test 命令

- 训练单帧 smoke：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/train.py --config temperal/configs/single_frame_smoke.yaml
  ```
- 评估单帧 smoke：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/evaluate.py --config temperal/configs/single_frame_smoke.yaml --checkpoint temperal/outputs/single_frame_smoke/last.pt
  ```
- 训练 GRU baseline：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/train.py --config temperal/configs/gru.yaml
  ```
- 评估 GRU baseline：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/evaluate.py --config temperal/configs/gru.yaml --checkpoint temperal/outputs/gru_exp/last.pt
  ```

### 更新后的关键结果（val split）

#### Single-frame baseline
- mean_l2: 0.38096
- median_l2: 0.36428
- success_at_0.2: 0.22667

#### CNN+GRU baseline
- mean_l2: 0.04079
- median_l2: 0.04017
- success_at_0.2: 1.0

#### Val split 常数基线（始终预测标签均值）
- mean_l2: 1.02017
- median_l2: 1.00575
- success_at_0.2: 0.09333

### 更新后的结论

- 相比最初“看最后几帧 + 全量评估”的设定，这一版任务明显更合理，也更接近 observe-then-act。
- 单帧模型已经明显优于常数基线，说明在前 `2/3` 轨迹中确实存在可学的预测信息。
- 但 CNN+GRU 仍然在验证集上接近满分，说明“预测最后一帧附近的低空位置”这个监督目标本身仍偏简单。
- 如果下一步想继续提高任务难度，优先建议修改监督目标：
  - 真正落点 / 截获点预测
  - 更早的 observation cutoff
  - time-to-impact 或 future rollout 预测

---

## 7) 2026-03-16 新数据 smoke：`20260316_204706`

这轮 smoke 使用了新采集的数据：

- 数据目录：`D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_204706`
- 数据设定：球下落范围和相机位置都做了调整，覆盖约 `120` 度扇面，半径约 `3-5m`，飞行时间约 `2.5-3s`
- 轨迹数量：`500`

### 数据概览

- 轨迹帧数均值：`140.718`
- 轨迹帧数范围：`127 - 152`
- 最后一帧 `ball_py` 均值：`0.30594`
- 最后一帧 `ball_py` 范围：`0.22808 - 0.37970`

相比 `20260314_140730` 的上一批数据，这批轨迹更长，视野变化更大。

### split 与配置

- split 目录：`temperal/data/splits_20260316_204706`
- train / val / test：`350 / 75 / 75`

#### single_frame_smoke_20260316.yaml
- observation_length: 8
- frame_stride: 2
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 1
- batch_size: 16
- lr: 1e-3

#### gru_20260316.yaml
- observation_length: 16
- frame_stride: 1
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 10
- batch_size: 8
- lr: 5e-4

### 抽样检查

- loader 抽样示例：`[0, 6, 12, 19, 25, 31, 37, 43, 50, 56, 62, 68, 74, 81, 87, 93]`
- 这表明模型只观察轨迹前 `67%` 的可见区间，并在其中均匀取样。

### 这轮 smoke test 命令

- 生成 split：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/make_splits.py --data_root D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_204706 --out_dir temperal/data/splits_20260316_204706
  ```
- 训练单帧 smoke：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/train.py --config temperal/configs/single_frame_smoke_20260316.yaml
  ```
- 评估单帧 smoke：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/evaluate.py --config temperal/configs/single_frame_smoke_20260316.yaml --checkpoint temperal/outputs/single_frame_smoke_20260316/last.pt
  ```
- 训练 GRU baseline：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/train.py --config temperal/configs/gru_20260316.yaml
  ```
- 评估 GRU baseline：
  ```bash
  d:\projects\rl-ball-fetch\rl_be\.venv\Scripts\python.exe temperal/scripts/evaluate.py --config temperal/configs/gru_20260316.yaml --checkpoint temperal/outputs/gru_exp_20260316/last.pt
  ```

### 关键结果（val split）

#### Single-frame baseline
- mean_l2: 0.78200
- median_l2: 0.44073
- success_at_0.2: 0.34667

#### CNN+GRU baseline
- mean_l2: 0.08680
- median_l2: 0.07184
- success_at_0.2: 0.93333

#### Val split 常数基线（始终预测标签均值）
- mean_l2: 2.30048
- median_l2: 2.51585
- success_at_0.2: 0.05333

### 与上一批数据的对比

- 新数据上的常数基线更差，说明标签分布显著变宽，任务本身更有区分度。
- 单帧模型从上一批数据的 `mean_l2 = 0.38096` 上升到 `0.78200`，说明单帧外观线索变得更不稳定。
- GRU 模型从上一批数据的 `mean_l2 = 0.04079` 上升到 `0.08680`，`success_at_0.2` 从 `1.0` 降到 `0.93333`，说明时序建模仍然很强，但任务已经不再接近“满分简单题”。

### 当前判断

- 这批新数据明显比上一批更难，也更接近你想要的研究问题。
- 但在“标签仍是最后一帧附近球位置”的前提下，GRU 依然表现非常强，说明任务还有继续变难的空间。
- 如果下一步再把监督目标换成真正落点 / 截获点，这个 benchmark 会更有说服力。

---

## 8) 2026-03-16 更难数据 smoke：`20260316_220723`

这轮 smoke 使用了更难的新数据：

- 数据目录：`D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260316_220723`
- 数据设定：`360` 度、半径 `0-6`
- 轨迹数量：`1000`

### 数据概览

- 轨迹帧数均值：`138.731`
- 轨迹帧数范围：`124 - 152`
- 最后一帧 `ball_py` 均值：`0.37499`
- 最后一帧 `ball_py` 范围：`0.16898 - 0.85965`

和前两批相比，这批数据在方位和半径上的覆盖都更激进，最后一帧高度分布也更散。

### split 与配置

- split 目录：`temperal/data/splits_20260316_220723`
- train / val / test：`700 / 150 / 150`

#### single_frame_smoke_20260316_220723.yaml
- observation_length: 8
- frame_stride: 2
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 1
- batch_size: 16
- lr: 1e-3

#### gru_20260316_220723.yaml
- observation_length: 16
- frame_stride: 1
- use_last_n_frames: false
- observation_end_fraction: 0.67
- sampling_mode: `uniform_visible`
- epochs: 10
- batch_size: 8
- lr: 5e-4

### 抽样检查

- loader 抽样示例：`[0, 6, 12, 18, 24, 30, 36, 42, 49, 55, 61, 67, 73, 79, 85, 91]`
- 这表明模型仍然只观察轨迹前 `67%` 的可见区间，并在其中均匀取样。

### 关键结果（val split）

#### Single-frame baseline
- mean_l2: 0.63026
- median_l2: 0.33193
- success_at_0.2: 0.30000

#### CNN+GRU baseline
- mean_l2: 0.15115
- median_l2: 0.12964
- success_at_0.2: 0.76000

#### Val split 常数基线（始终预测标签均值）
- mean_l2: 1.76590
- median_l2: 1.36297
- success_at_0.2: 0.06000

### 与前两批数据的对比

- 相比 `20260316_204706`，GRU 的 `mean_l2` 从 `0.08680` 上升到 `0.15115`，`success_at_0.2` 从 `0.93333` 降到 `0.76000`。
- 相比 `20260314_140730`，GRU 已经明显不再接近满分，说明这批数据把任务进一步拉难了。
- 常数基线依然很差，说明标签分布本身并不集中，模型确实需要利用视觉时序信息。

### 当前判断

- 这批 `360` 度、半径 `0-6` 的数据是目前三批里最接近“非简单题”的一版。
- GRU 仍然明显优于单帧，说明时序信息依然很关键。
- 但因为监督目标仍是“最后一帧附近球的 `(x, y)` 位置”，任务还不是最终形态；如果改成真正落点 / 截获点，难度和研究价值还会进一步上去。
