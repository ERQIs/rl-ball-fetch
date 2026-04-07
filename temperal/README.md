# Temperal: Observe-Then-Act Supervised Benchmark (scaffold)

Minimal scaffold to run supervised landing-point regression experiments from Unity-exported episodes.

Layout
- `configs/` example YAML configs
- `src/datasets/trajectory_dataset.py` dataset loader
- `src/models/` models (single-frame, CNN+GRU)
- `src/engine/` training/evaluation loops + metrics
- `scripts/` utilities to train/evaluate/make_splits

Quickstart

1) Install dependencies:

```bash
pip install -r temperal/requirements.txt
```

2) (Optional) generate splits:

```bash
python temperal/scripts/make_splits.py --data_root D:/projects/rl-ball-fetch/ball_fetch/vis_backbone/datasets/manual_capture/20260314_140730 --out_dir temperal/data/splits
```

3) Train (example):

```bash
python temperal/scripts/train.py --config temperal/configs/single_frame.yaml
```

4) Evaluate:

```bash
python temperal/scripts/evaluate.py --config temperal/configs/single_frame.yaml --checkpoint temperal/outputs/single_frame_exp/last.pt
```

Notes
- The dataset expects per-episode `frames/` and `trajectory.csv` with ball position columns; the loader heuristically finds x/y columns and uses the last row as landing point.
- Configs specify `observation_length`, `frame_stride`, and `use_last_n_frames`.
- This scaffold is intentionally minimal; extend `trajectory_dataset.py` for custom csv formats.
下面给出一份更正式、面向实验执行与协作开发的技术文档版本。该版本明确去除 few-shot adaptation 作为当前主叙事，将第二阶段工作的核心重新收束为：
# Part II Research Plan v1

## Temporal Visual Representation for Structured Spatiotemporal Capability Enhancement and Data-Efficient Learning in Mobile Ball Catching

---

## 1. Scope and Positioning

### 1.1 Context

第一阶段工作已经初步验证：在移动接球任务中，引入显式**空间连续性先验**的视觉头，有助于提升下游策略学习的效率。这一结果表明，视觉接口的结构偏置会显著影响 embodied learning 的 sample efficiency。由此可进一步提出一个更强的问题：若任务本质上依赖对目标运动的理解，则除了空间结构之外，**时序结构应如何被编码**，以及这类编码是否能够进一步提升学习效率与下游性能。

### 1.2 Shift of Narrative

当前阶段不再沿用 few-shot adaptation 作为主叙事。原因在于，对于当前接球任务，few-shot adaptation 所依赖的“在线反馈—误差感知—快速修正”链条尚不清晰。尤其在 observe-then-act 或高速拦截设定下，环境很难提供一种低成本、局部且可重复利用的 adaptation signal，用于明确指示当前动作与最优解之间的偏差程度。因此，若在当前阶段强行围绕 few-shot adaptation 组织故事，会导致问题定义含混、反馈机制不清、实验目标不稳定。

基于这一判断，当前工作主线改为：

> 研究何种 temporal visual representation 及其预训练机制，能够增强模型的结构化时空表征能力，并提升接球任务中的 data-efficient learning。

### 1.3 Research Focus

Part II 的关注点不是泛泛地“使用时序信息”，而是系统考察以下三个维度是否能增强时空能力：

1. **Temporal architecture design**
2. **Visual-temporal pretraining**
3. **Continuity-based visual priors in the encoder**

这三者共同构成 Part II 的核心变量空间。

---

## 2. Problem Statement

### 2.1 Top-Level Question

在移动接球任务中，如何构建一种视觉—时序表征，使其能够从有限长度的观测序列中提取对控制有用的运动信息，并在下游任务中体现出更高的数据效率？

### 2.2 Operational Reformulation

为了使该问题可被系统研究，需要将其操作化为如下形式：

* 输入为一段有限长度的图像观测序列；
* 模型通过视觉编码与时序聚合得到一个 latent representation；
* 该表示应编码与接球决策相关的动态量，如轨迹趋势、到达时刻、未来截获位置等；
* 该表示最终服务于下游学习任务，并在固定数据预算下表现出更好的收敛速度、更强的样本效率或更高的最终性能。

因此，Part II 的主问题可以进一步写为：

> Which temporal visual representation most effectively extracts interception-relevant motion information from short visual sequences, and how do architecture, pretraining, and continuity priors affect its downstream data efficiency?

---

## 3. Controlled Task Simplification

### 3.1 Motivation

在原始闭环接球任务中，视觉变化同时来源于：

* 球的外在运动；
* 小车自身动作造成的 ego-motion；
* 由两者耦合产生的相对运动模式。

若直接在该设定下研究 temporal representation，则需要同时处理 motion understanding、ego-motion disentanglement 和 policy learning，问题复杂度过高，不利于隔离时序表征本身的作用。

### 3.2 Simplified Setting: Observe-Then-Act

因此，当前阶段采用如下受控简化设定：

1. 小车在观察阶段保持静止；
2. 在时长为 (T_{\text{obs}}) 的观察窗口内接收球的图像序列；
3. 观察结束后，模型输出一个二维速度向量 (a \in \mathbb{R}^2)；
4. 小车以该恒定速度执行固定时长 (T_{\text{act}})；
5. 根据最终是否成功接球，或根据终端位置误差评估任务表现。

### 3.3 Value of This Setting

该简化设定具有以下优点：

* 将“时序视觉是否学到运动信息”从“车体运动补偿”中剥离出来；
* 使 temporal representation 的分析更干净；
* 更适合构造监督信号；
* 更适合开展 architecture / pretraining / prior 的系统消融；
* 为后续回到 fully closed-loop control 提供一个中间研究台阶。

该设定应被视为原始任务的受控子问题，而非对原问题的替代。其目标在于先回答：**在没有 ego-motion 干扰的条件下，模型能否构建对接球有用的 motion-aware representation。**

---

## 4. Action Parameterization and Supervision Structure

### 4.1 Action Definition

当前简化任务中，动作定义为一个二维速度向量：

[
a = (v_x, v_y)
]

并在固定时长 (T_{\text{act}}) 内恒定执行。

### 4.2 Equivalence to Terminal Target

在以下条件成立时：

* 小车起始位置固定；
* 球的发射位置固定；
* 动作执行时长固定；
* 动作是理想恒定速度；

则动作 (a) 与终端目标位置 (p^\star) 之间存在确定性线性关系：

[
p^\star = p_0 + a \cdot T_{\text{act}}
]

因此，在该设定下，**动作预测**与**终端截获位置预测**是同一问题的两种参数化形式。严格来说，更适合作为监督目标的不是“球落地点”本身，而是与任务定义一致的**terminal interception location** 或与接球终态相关的目标平面位置。

### 4.3 Implication

这一点意味着，下游任务可以设计为两类形式：

* **Sequence-to-action**：输入观测序列，直接输出速度；
* **Sequence-to-target**：输入观测序列，输出目标截获位置，再解析映射为动作。

第二种方式通常更有解释性，因为它更直接刻画模型是否真正恢复了与轨迹相关的动态量。

---

## 5. Core Research Axes

Part II 的实验主线围绕三个变量展开。

### 5.1 Axis A: Temporal Architecture

研究问题：不同 temporal architecture 对时空结构建模能力有何差异？

候选模型包括但不限于：

* GRU / RNN-based temporal encoder
* ConvGRU-based spatiotemporal encoder
* 老板已有时序代码所对应的 structured recurrent architecture
* 其他具有显式状态滚动机制的 temporal latent model

该主轴主要回答：**哪种 temporal state organization 更适合接球任务中的 motion extraction。**

### 5.2 Axis B: Pretraining

研究问题：在进入下游任务之前，是否应先利用 trajectory data 做 visual-temporal pretraining？

核心思想是：先用无控制干预或弱控制干预的轨迹数据，训练视觉—时序模型获得更有结构的 latent state，再将其迁移到下游任务。该主轴主要回答：**temporal pretraining 是否提升下游 data efficiency。**

### 5.3 Axis C: Continuity-Based Visual Prior

研究问题：第一阶段已验证有效的 continuity-based spatial encoder，是否能作为 temporal model 的更优输入接口？

这将测试：**空间连续性先验是否不仅在单帧 setting 有效，而且能够增强 temporal representation 的学习与使用。**

---

## 6. Research Hypotheses

当前阶段可明确提出以下工作假设：

### H1

在 observe-then-act 的接球任务中，具有显式状态聚合能力的 temporal encoder 能够形成比弱时序聚合方式更有效的 motion-aware representation。

### H2

在 trajectory data 上进行 visual-temporal pretraining，可以增强 latent state 中的结构化动力学信息，并提升下游任务的数据效率。

### H3

continuity-based visual encoder 作为 temporal input interface 时，能够提供更稳定、更结构化的空间表示，从而进一步增强 temporal model 的性能。

### H4

temporal architecture、pretraining 与 continuity prior 之间可能存在交互作用；即某些 temporal model 对预训练或 continuity prior 的收益更敏感。

---

## 7. Downstream Task Design

当前阶段明确放弃 few-shot adaptation 作为主评测目标，改为采用**representation-oriented downstream tasks** 与 **control-oriented downstream tasks** 的分层设计。

### 7.1 Layer 1: Representation-Oriented Downstream Tasks

该层任务用于评估 temporal representation 是否成功编码了与运动相关的关键量。可采用如下监督目标：

#### 7.1.1 Interception Target Prediction

根据观察序列预测终端截获位置或目标对齐位置。该任务直接测试模型是否恢复了对接球最相关的空间—时间目标。

#### 7.1.2 Time-to-Impact Prediction

预测球到达某个参考平面或接球时刻的剩余时间。该任务测试模型是否理解时间动态。

#### 7.1.3 Velocity / Trajectory Parameter Prediction

预测球当前速度、拟合的轨迹参数，或某种低维 dynamics descriptor。该任务测试模型是否真正提取了运动状态。

#### 7.1.4 Future Position Prediction

预测未来若干离散时刻的球位置序列。该任务更接近老板时序代码中的“latent rollout supports future prediction”思想。

### 7.2 Layer 2: Control-Oriented Downstream Tasks

该层任务用于评估 representation 是否转化为实际控制收益。

#### 7.2.1 Action Regression

输入观察序列，输出二维速度动作。标签可由已知物理量解析获得，或由 oracle controller 产生。

#### 7.2.2 RL-Based Catching Evaluation

将 pretrained visual-temporal representation 接入 RL policy，考察其 sample efficiency、最终成功率与终端误差。该层不作为第一优先级主评测，而作为 representation 结论的控制层验证。

### 7.3 Recommended Priority

当前建议的优先级为：

1. **Representation-oriented supervised tasks as the main evaluation platform**
2. **Action regression as a secondary bridge task**
3. **RL evaluation as additional downstream validation**

原因在于：监督任务更适合 isolating representation quality，而 RL 更适合用来验证这些表示是否真正有助于 embodied control。

---

## 8. Pretraining Design

### 8.1 Objective of Pretraining

预训练的目标不是直接学习最终控制策略，而是为视觉—时序模型提供更强的结构化时空能力，包括：

* 空间结构保留；
* 运动一致性编码；
* 动力学相关 latent state 形成；
* 对未来演化的可预测性。

### 8.2 Candidate Pretraining Signals

预训练可以选择以下一种或多种辅助目标：

#### 8.2.1 Future Observation Prediction

要求 latent state 支持未来帧或未来特征的预测。适用于测试 temporal state 是否携带 dynamics information。

#### 8.2.2 Motion-State Regression

要求模型预测 position、velocity、time-to-impact 或 trajectory parameters。适用于构建具有明确物理意义的 state。

#### 8.2.3 Continuity / Transport Auxiliary Loss

若视觉头采用 continuity-based encoder，可继续施加 feature transport consistency、warp-decoding consistency 等损失，从而将第一阶段的空间先验延续到时序 setting 中。

#### 8.2.4 Reconstruction-Based Regularization

通过 reconstruction loss 避免 latent collapse，并强制 latent 保留可解码的结构信息。

### 8.3 Implementation Principle

预训练目标不宜过多。建议从一个主目标与少量辅助目标开始，避免系统复杂度快速膨胀，导致实验解释困难。推荐初始方案为：

* temporal latent state construction
* one motion-related regression objective
* optional continuity-based auxiliary loss if continuity encoder is used

---

## 9. Experimental Design

### 9.1 Factorized Experimental Structure

Part II 的整体实验可视为一个三因素设计：

[
\text{Performance} = f(\text{temporal architecture}, \text{pretraining}, \text{continuity prior})
]

其中：

* temporal architecture：GRU / ConvGRU / structured recurrent model / other baselines
* pretraining：with vs without pretraining
* continuity prior：plain visual head vs continuity-based visual head

### 9.2 Core Ablation Questions

所有实验应围绕如下核心问题展开：

#### Q1

不同 temporal architecture 中，哪一种最适合提取 interception-relevant motion information？

#### Q2

在相同 temporal architecture 下，pretraining 是否显著改善下游任务性能与数据效率？

#### Q3

continuity-based visual encoder 是否在 temporal setting 中依旧有效？

#### Q4

architecture、pretraining 与 continuity prior 是否存在交互作用？

### 9.3 Experimental Progression

建议按如下顺序推进：

#### Stage A: Build Controlled Supervised Benchmark

先固定 observe-then-act setting，建立 sequence-to-target / sequence-to-motion 的监督学习 benchmark。

#### Stage B: Compare Temporal Architectures

在无预训练、统一视觉头条件下比较 temporal models，获得纯 architecture 层结论。

#### Stage C: Add Pretraining

在表现较好的 temporal models 上加入 visual-temporal pretraining，比较数据效率与最终性能变化。

#### Stage D: Add Continuity Prior

将 plain encoder 替换为 continuity-based encoder，观察 temporal downstream 是否进一步提升。

#### Stage E: Optional RL Transfer

将最优 representation 迁移到 RL catching policy 中，验证其对实际控制的贡献。

---

## 10. Metrics

### 10.1 Representation-Oriented Metrics

* interception target prediction error
* time-to-impact error
* velocity estimation error
* future position prediction error
* calibration curve or uncertainty-quality relation if available

### 10.2 Control-Oriented Metrics

* action regression error
* catch success rate
* terminal miss distance
* sample efficiency curve under fixed data budget
* training stability across seeds

### 10.3 Data-Efficiency Metrics

由于主叙事之一是 data-efficient learning，因此所有关键实验应报告：

* 在不同训练样本数下的性能曲线
* 达到某一性能阈值所需的样本量
* 小数据 regime 下不同模型之间的相对优势

即，Part II 的结果应尽量避免只报告最终收敛性能，而应突出**在有限数据预算下谁学得更快、更稳、更有效**。

---

## 11. Why RL Is Not the Primary Evaluation at This Stage

尽管接球最终是控制任务，但当前阶段不宜将 RL 作为唯一主评测。原因如下：

1. RL 中存在大量与表征设计无关的噪声源；
2. reward shaping 对结果影响过大，容易掩盖 representation 的真实作用；
3. 不同模型之间 optimization stability 的差异会污染 architecture / pretraining 结论；
4. 当前模拟环境已提供足够丰富的监督标签，先做监督评测成本更低、结论更干净。

因此，Part II 的技术路线应明确区分：

* **representation evaluation first**
* **control validation later**

---

## 12. Excluded Narrative: Why Few-Shot Adaptation Is Deferred

当前版本明确不将 few-shot adaptation 作为 Part II 主任务，原因总结如下：

1. 接球任务缺乏清晰、局部且稳定的 adaptation feedback；
2. 目前尚未形成一个可操作的“偏离最优解程度”在线信号；
3. 若直接纳入 few-shot adaptation，会使主变量从 representation 扩展到 feedback design、adaptation protocol 与 online calibration，问题规模显著膨胀；
4. 当前阶段更有价值的问题是：**先构造一个结构化时空表征，再验证其是否支持 data-efficient downstream learning。**

因此，few-shot adaptation 并未被理论上否定，而是被策略性后置，待 representation 与 downstream benchmark 更稳定后再考虑回接。

---

## 13. Expected Contribution of Part II

若上述技术路线成立，则 Part II 预计能形成如下贡献类型：

### 13.1 Methodological Contribution

提出并系统比较若干增强结构化时空能力的机制，包括 temporal architecture、visual-temporal pretraining 与 continuity-based visual priors。

### 13.2 Empirical Contribution

在移动接球的受控时序任务中，系统评估不同视觉—时序表示在小数据 regime 下的性能与样本效率。

### 13.3 Structural Contribution

将第一阶段的空间连续性表征工作自然延伸到第二阶段的时序建模，形成“spatial prior → temporal representation → data-efficient downstream learning”的连续研究链条。

---

## 14. Immediate Technical Agenda

基于当前主线，下一步技术工作建议按以下顺序执行：

### 14.1 Finalize Task Definition

明确 observe-then-act 设定中的：

* (T_{\text{obs}})
* (T_{\text{act}})
* observation frequency
* action bounds
* success criterion

### 14.2 Finalize Primary Supervised Targets

确定第一版主要标签，例如：

* terminal interception location
* time-to-impact
* ball velocity / trajectory descriptor

### 14.3 Build Baseline Temporal Models

实现第一批 temporal baselines，并统一输入/输出接口。

### 14.4 Define Pretraining Protocol

选定 trajectory pretraining 数据、主目标与辅助损失。

### 14.5 Build Factorized Ablation Matrix

明确 architecture × pretraining × continuity prior 的实验表格，并建立统一训练评测脚本。

### 14.6 Reserve RL Transfer Slot

仅为后续控制验证预留接口，不在第一轮实验中优先投入。

---

## 15. Compact Working Summary

当前阶段的正式工作线可概括为：

> Part II studies how to enhance structured spatiotemporal capability in visual representations for mobile ball catching. The central goal is to identify temporal architectures, pretraining strategies, and continuity-based visual priors that improve motion-aware representations and lead to more data-efficient downstream learning. To isolate temporal understanding from ego-motion, the task is reformulated into an observe-then-act setting, where a stationary observation window is followed by a single fixed-velocity action. The primary evaluation platform will be supervised downstream tasks that probe motion-relevant prediction and action generation, while reinforcement learning will be treated as a secondary control-oriented validation.

---

如果需要，下一步可以继续把这份文档往更工程化方向推进，整理成下面两种形式之一：

1. **实验执行版**：加入具体实验表格、变量枚举、目录结构与代码模块划分。
2. **论文草案版**：改写成更接近 proposal / methods section 的学术文本。
