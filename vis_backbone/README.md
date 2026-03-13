# Stage A 视觉表征学习说明

## 1. 目标

构建一个最小可复现的自监督视觉表征模块，使特征具备：

- 空间连续性
- 运动一致性（可随光流搬运）
- 可解码性（避免塌缩）

输出应是带空间结构的 feature map，而不是单一全局向量。

## 2. 输入输出接口

输入样本：

- `I_t`: 当前帧 `(3, 64, 64)`
- `I_t1`: 下一帧 `(3, 64, 64)`
- `flow_t`: `I_t -> I_t1` 的 dense flow `(2, 64, 64)`

模型输出：

- `F_t = E(I_t)`，`F_t1 = E(I_t1)`，shape `(C, 8, 8)`
- `I_hat_t = D(F_t)`
- `F_warp = Warp(F_t, flow_t)`
- `I_hat_t1_from_warp = D(F_warp)`

## 3. 模型结构

Encoder（CNN）：

- 逐步下采样到 `8x8`
- 保留空间结构，不做全局池化

Decoder（轻量上采样网络）：

- 从 `8x8` 逐步恢复到 `64x64`
- 用于 anti-collapse，不追求生成质量

Warp 模块：

- 将 flow 映射到 feature grid
- 使用 `grid_sample` 做双线性采样

## 4. Loss 设计

总损失：

- `L_total = λ_rec*L_rec + λ_trans*L_trans + λ_wd*L_wd (+ λ_nb*L_nb)`

三项主损失：

- `L_rec`: 当前帧重建损失（保信息）
- `L_trans`: warp 后 feature 与下一帧 feature 对齐（保运动一致）
- `L_wd`: warp 后解码重建下一帧（保语义一致）

可选辅助项：

- `L_nb`: 邻域平滑约束

## 5. 训练流程

单步流程：

1. 编码 `I_t` 与 `I_t1`
2. 重建 `I_t`
3. 用 flow warp `F_t`
4. 解码 `F_warp` 得到下一帧预测
5. 计算并加权三项 loss，反向传播

## 6. 默认超参建议

- `image_size=64`
- `feature_size=8`
- `C=64`
- `batch_size=32`
- `optimizer=Adam`
- `lr=1e-3`
- `λ_rec=1.0`
- `λ_trans=1.0`
- `λ_wd=1.0`
- `λ_nb=0.0`

## 7. 最小消融实验

- AE baseline: 仅 `L_rec`
- Transport only: 仅 `L_trans`
- Warp-decode only: 仅 `L_wd`
- Full: `L_rec + L_trans + L_wd`

## 8. 工程注意事项

- 必须做 warp 单元测试（常量平移 case）
- flow 与 feature grid 坐标尺度必须一致
- 第一轮先用轻量 CNN，不急于上复杂 backbone
- decoder 不要过强，避免补偿掩盖编码缺陷

## 数据采集：

环境里在小车上挂了一个data collector，配好手动跑就能采数据。