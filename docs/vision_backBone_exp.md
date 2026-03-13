# 空间先验视觉 Backbone 在简单 RL 环境中的实验设计

## 1. 实验目标

验证两件事：

1. **带空间先验的视觉 backbone 是否比 global backbone 更容易用 RL 训练。**
2. **先做 continuity / transport 预训练，是否能让后续 RL 更稳定、更省样本。**

这里的核心不是只看最终 reward，而是比较不同视觉表示对下游控制学习的**样本效率、稳定性和可重用性**。

---

## 2. 核心假设

### H1：空间先验有助于从零开始的 RL 训练

相较于将整张图很快压成单个全局向量的 global backbone，保留 feature map 空间结构的 backbone 更容易学到对控制有用的表示，因此训练更稳定、样本效率更高。

### H2：预训练过的空间 backbone 能提供更好的视觉初始化

如果 backbone 先通过 continuity / transport 学到了更规整的视觉表示，那么后续 RL 应该更快进入有效学习阶段。

### H3：预训练表征本身具有一定可读性

如果预训练确实学到了有结构的表示，那么即使冻结 backbone，只训练 action head，也应当能达到可用性能，或者至少比从零开始更快。

---

## 3. Backbone 定义

### Global backbone

* 输入图像后，较早压缩为单个 **global vector**。
* 不显式保留二维空间格子结构。
* 典型形式：CNN + global average pooling + MLP head，或 pooled token + MLP head。

### Spatial backbone

* 输入图像后输出 **feature map**（例如 `16x16` 或 `8x8`）。
* 显式保留局部空间结构与邻接关系。
* 可选地用 continuity / transport 目标进行预训练。

要求：两类 backbone 尽量保持参数量和 head 容量接近，避免容量差异成为主要解释。

---

## 4. 实验设置

### E1. Global-Scratch-E2E

**目的**：作为标准视觉 RL 基线。

* backbone：global backbone
* 初始化：随机初始化
* 训练方式：端到端 RL
* 可训练参数：backbone + action head

**回答问题**：没有空间先验时，从零开始 RL 能训到什么水平？

### E2. Spatial-Scratch-E2E

**目的**：测试空间先验架构本身是否更好训。

* backbone：spatial backbone
* 初始化：随机初始化
* 训练方式：端到端 RL
* 可训练参数：backbone + action head

**回答问题**：仅靠空间结构先验，是否已经比 global backbone 更稳定、更省样本？

### E3. Spatial-Pretrain-Frozen

**目的**：测试预训练表征本身的可读性和可重用性。

* backbone：spatial backbone
* 初始化：先做 continuity / transport 预训练
* 训练方式：冻结 backbone，只训练 action head
* 可训练参数：action head

**回答问题**：预训练后的视觉表示，是否已经足够支撑控制，不需要 RL 再改视觉？

### E4. Spatial-Pretrain-Finetune

**目的**：测试预训练是否给 RL 提供更好的初始化。

* backbone：spatial backbone
* 初始化：先做 continuity / transport 预训练
* 训练方式：端到端 RL finetune
* 可训练参数：backbone + action head

**回答问题**：预训练是否能让后续 RL 更快、更稳？

---

## 5. 关键比较关系

### C1. E1 vs E2

比较 **global vs spatial** 在从零开始 RL 时的差异。

**结论含义**：

* 若 E2 优于 E1，说明空间先验架构本身有助于训练。

### C2. E2 vs E4

比较 **scratch spatial** 与 **pretrained spatial finetune**。

**结论含义**：

* 若 E4 优于 E2，说明 continuity / transport 预训练确实为 RL 提供了更好的视觉初始化。

### C3. E3 vs E4

比较 **冻结预训练表征** 与 **预训练后继续端到端 finetune**。

**结论含义**：

* 若 E3 已经很好，说明预训练表征本身很强、很容易被 head 读取。
* 若 E4 明显优于 E3，说明预训练主要提供了好初始化，但 RL 继续适配视觉仍然重要。

### C4. E1 vs E4

比较最弱基线与最佳预期方案。

**结论含义**：

* 用于展示整体最优路线是否成立，但不作为最主要机制对照。

---

## 6. 评估指标

至少记录以下指标：

* **Return vs environment steps**：主学习曲线
* **Success rate vs environment steps**：更直观地看任务完成情况
* **达到固定成功率阈值所需步数**：直接反映样本效率
* **多随机种子均值与方差**：反映训练稳定性
* **最终成功率 / 最终 return**：反映性能上限

建议至少使用多个随机种子，避免单次曲线误导结论。

---

## 7. 结果解释原则

### 如果 E2 > E1

说明空间先验架构本身就有帮助。

### 如果 E4 > E2

说明预训练学到的连续性 / 运动一致性表示对 RL 有额外帮助。

### 如果 E3 接近 E4

说明预训练表征本身已经足够好，RL 主要是在学习 action head。

### 如果 E4 明显高于 E3

说明预训练给了好的起点，但端到端 RL 对视觉表示继续适配仍然必要。

### 如果 E3 很差但 E4 很好

说明预训练表征并不能直接被简单 head 读出，但它仍然是一个比随机初始化更好的优化起点。

---

## 8. 风险与控制变量

### 主要风险

1. **模型容量不匹配**：会导致结果难解释。
2. **任务过于简单**：可能所有方法都很快收敛，看不出差异。
3. **任务过难或目标过小**：视觉分辨率瓶颈会掩盖 backbone 差异。
4. **seed 波动大**：容易把偶然训练成功误当成方法优势。

### 控制变量

* 保持 RL 算法、超参数、batch size、训练步数一致
* 尽量匹配 backbone 参数量和 action head 容量
* 使用相同环境与奖励设计
* 使用相同随机种子集合进行比较

---

## 实验结果

选取了e1_60_3_301_agr_reward环境。实验数据是

rl_be\results\exp_vb_0310\E1\exp_vb_0310_E1_easy60_agr_build

rl_be\results\exp_vb_0310\E2\exp_vb_0310_E2_easy60_agr_build

rl_be\results\exp_vb_0310\E3\exp_vb_0310_E3_easy60_agr_build

rl_be\results\exp_vb_0310\E4\exp_vb_0310_E4_easy60_agr_build