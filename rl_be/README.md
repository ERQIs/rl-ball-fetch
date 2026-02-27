# ball_fetch 训练后端（ML-Agents）

## 当前用的是什么模型

- 算法：`PPO`
- 网络：`MLP`（2 层隐藏层，每层 128 单元）
- 配置文件：[catch_ppo.yaml](d:\erqi\bishe\exp\unity\proj\rl_be\config\catch_ppo.yaml)

对应 Unity 行为名是 `CarCatch`。

## 为什么会在 50000 step 报错退出

你报错的根因是：

- 训练到 `checkpoint_interval: 50000` 时会触发一次模型导出（ONNX）。
- 你当时环境里是 `torch 2.10.0`，它的 ONNX 导出链路会依赖 `onnxscript`。
- `mlagents 1.1.0` 的依赖组合并不适配这条新导出链路，所以在导出时崩掉。

现在已经修成稳定组合：

- `mlagents==1.1.0`
- `torch==2.1.2`
- `onnx==1.15.0`
- `protobuf==3.20.3`
- `numpy==1.23.5`

## 首次或重建环境

```powershell
.\scripts\setup_env.ps1
```

## 在 Unity Editor 里训练

1. 打开 `CatchCarScene`
2. 点 `Play`
3. 运行：

```powershell
.\scripts\train_editor.ps1 -RunId carcatch_v1 -TimeoutWait 300
```

## 从上次中断点继续训练

如果你的 `results\carcatch_v1` 已有 checkpoint：

```powershell
.\scripts\train_editor.ps1 -RunId carcatch_v1 -Resume -TimeoutWait 300
```

## TensorBoard

```powershell
.\scripts\tensorboard.ps1
```

打开：`http://localhost:6006`
