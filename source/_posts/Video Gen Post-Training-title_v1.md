---
title: Video Gen Post-Training 选题分析
date: 2026-03-20
tags:
  - 生成模型
  - Transformer
  - Diffusion
  - Video Generation
  - Post-Training
categories:
  - 博士科研
index_img: /img/Video gen post-training.png
math: true
---

# Video Gen Post-Training 选题分析

按照研究方法论中的选题步骤：精确化 failure case → 判断 well-established solution 归属（4种情况）→ 确定技术挑战 → 选题决策。

---

## 选题判断框架（来自方法论）

| 情况 | 描述 | 建议 |
|---|---|---|
| 情况1 | 相同I/O任务，已有不错解决方案，只是有些地方还不够好 | **必须换题** |
| 情况2 | I/O变化/多个不同域，都有不错的类似解决方案 | **必须换题** |
| 情况3 | 只有少数1-2个完全不同域有不错的类似解决方案 | 适合新手做 |
| 情况4 | 各个不同领域都没有较好的解决方案 | 适合高手做 |

---

## Gap B 分析：复杂多步骤因果链物理

### Failure Case（精确化）

**任务设定**：Text-to-Video，prompt 中包含多步物理因果链

**具体 failure case**：
- Prompt：*"A ball rolls down a ramp and hits a domino, which falls and knocks over a cup of water"*
- 失败模式1：球碰到多米诺时，多米诺没有物理反应（因果链断裂）
- 失败模式2：多米诺倒了，但没有和杯子交互（中间环节断裂）
- 失败模式3：每个单独时刻的物理是对的（球在滚/多米诺在倒/杯子在倒），但前后缺乏因果连接

**与已有工作的本质区别**：
- PhysMaster/PhysCorr/PhyGDPO 解决的是：**单个对象在单一时刻的物理合理性**（球落地时的轨迹对不对）
- 本 failure case 要求：**步骤A的结果是步骤B发生的物理原因**（因果性，而非单步合理性）

### Well-established Solution 判断

| 领域 | 是否有类似解决方案 | 说明 |
|---|---|---|
| Video gen post-training | ❌ | 当前所有物理对齐工作仅处理单对象/单时刻，无因果链 |
| 物理模拟引擎 | ✅ 但不适用 | Physics engine 可模拟因果，但不是生成模型，无法直接迁移 |
| 机器人/具身AI | ⚠️ 部分 | 有因果场景理解，但未用于生成模型后训练 |
| 视频理解 | ⚠️ 部分 | 因果事件关系推理（视频问答），但没有作为reward用于生成 |
| 图像/LLM | ❌ | 无时序因果维度 |

**判断：情况4** — 各个不同领域的生成模型中，都没有针对多步骤物理因果链的后训练解决方案。

### 技术挑战

1. **Reward 设计极难**：如何衡量"A 导致了 B"而不是"A 和 B 各自合理"？需要因果关系度量，而非独立物理度量。
2. **标注困难**：人工标注因果链正确性非常专业，VLM 对复杂物理因果推理能力有限。
3. **数据稀缺**：包含多步骤物理因果链的真实视频数量有限。

### 选题决策：**可以做，但适合高手**

**风险极高，但影响力极大**。适合已有深厚物理推理/reward design 积累的研究者。  
不建议作为第一个项目，除非有明确的reward设计突破口。

---

## Gap C 分析：时序语义细粒度对齐

### Failure Case（精确化）

**任务设定**：Text-to-Video，prompt 中包含明确的时序语义结构（顺序动作/并发事件）

**具体 failure case**：

**Case C1：动作顺序错误**
- Prompt：*"First, a person opens a door. Then, they walk through it. Finally, they sit down at a desk."*
- 失败模式：视频中人物直接坐下（最后动作先发生），开门动作出现在末尾或缺失

**Case C2：动作重叠/遗漏**
- Prompt：*"The chef first chops vegetables, then stirs them in the pan, and finally plates the dish."*
- 失败模式：切菜和翻炒同时进行（时序边界不清）；或"摆盘"动作完全缺失

**Case C3：时序结构丢失**
- Prompt 中有 "while A, B happens" 的并发时序
- 失败模式：A 和 B 被顺序生成而非并发，视频时序结构完全不符合 prompt 语义

**与已有工作的本质区别**：

| 已有工作 | 解决的是 | 不解决的是 |
|---|---|---|
| VideoDPO | 全局 CLIP 语义相似度（视频整体和 prompt 相关不相关）| 时序内部的语义结构 |
| DenseDPO | 时序段的视觉运动质量（流畅度、artifact）| 这段时间里"发生了什么语义动作" |
| LocalDPO | 时空区域的视觉细节质量 | 语义动作的时序顺序 |
| RealDPO | 动作的自然流畅性（motion realism）| 多步骤动作的时序语义正确性 |

**核心技术问题**：如何构建能够捕获「动作时序语义正确性」的偏好对？

### Well-established Solution 判断

| 领域 | 是否有类似解决方案 | 说明 |
|---|---|---|
| Video gen post-training | ❌ | DenseDPO 时序段 = 运动质量；无人做时序语义 |
| 视频理解/时序动作识别 | ✅ 部分可迁移 | Temporal action segmentation（THUMOS/ActivityNet 等）可以识别"何时发生了什么动作" |
| LLM 时序推理 | ✅ 文本域 | Chain-of-thought 可以做时序语义推理，但不是视觉 |
| 图像生成对齐 | ⚠️ 弱相关 | LocalDPO 做了空间语义区域，但无时序 |
| T2I compositional | ⚠️ 弱相关 | Attend-and-Excite 等做了空间组合，无时序 |

**判断：情况3** — 视频理解领域有时序动作识别的技术基础，但没有将其作为生成模型后训练 reward 的工作；与之相近的域（LLM 的时序推理）也仅在文本域成熟，视觉域无对应。

### ✅ Failure Case 真实性验证（Benchmark 定量证据）

**强有力证据**（搜索确认）：

**TC-Bench** (2406.08656, ACL Findings 2025)：
- 专门测试 T2V 模型的 Temporal Compositionality（时序组合性）
- 测试类别：**属性渐变**（A→B状态转变）、**物体关系变化**、**背景切换**
- 关键结论：**"most video generators achieve less than 20% of the compositional changes"**
- 这是对 Sora、Gen-2 等 SOTA 模型的测试结果，说明失败率 >80%

**T2V-CompBench** (2407.14505, CVPR 2025)：
- 测试类别包括：**consistent attribute binding, dynamic attribute binding, spatial relationships, motion binding, action binding, object interactions**
- 关键结论："compositional text-to-video generation is **highly challenging** for current models"
- VideoDPO 论文本身也承认："such as those for **compositional video generation**" 是尚未解决的挑战

**TemporalBench** (2410.10818)：
- 测试 VideoLLM 的时序理解（包括 event order 评估）
- 关键结论：**GPT-4o 只达到 38.5% accuracy**，humans 约 70%，gap 约 30%——说明即使理解模型也有严重的时序推理缺陷，生成模型只会更差

**后训练领域的空白验证**：
- 专门搜索"video generation post-training temporal compositionality sequential actions DPO GRPO"→ **零命中**
- "Temporal Preference Optimization for Long-Form Video Understanding" (2501.13919) 是 VideoLLM 的理解优化，非视频生成后训练

**结论：Failure case 真实且重要，无对应生成后训练工作，确认为开放问题。**

### TC-Bench vs Gap C 的关系（重要细化）

TC-Bench 测试的是 **2-state 转变**（初始状态 → 结束状态），如"球从红色变蓝色"。  
Gap C 目标的是 **N-step 顺序动作**（A → B → C），如"先开门，再走进去，最后坐下"。

TC-Bench 是 Gap C 的 **子集**（最简单的 N=2 情况）且 <20% 成功率——意味着更复杂的 N≥3 情况更差，问题更严峻。

### 技术挑战（真实的，不是trivia）

**挑战1：偏好对构建方法**
- DenseDPO 用 corrupted GT 视频构建 pairs（motion bias 中立）
- 时序语义对 pair 如何构建？两个视频必须"内容相似但顺序不同"
- 方案A：生成多个 candidate，用 VLM 评估哪个时序语义更正确（成本高）
- 方案B：通过时序打乱（temporal shuffling）构造负样本（技术上有难点：如何保持视觉质量的同时改变语义顺序？）
- 方案C：将正确顺序视频的帧重排成错误顺序作为负样本（简单但不现实）

**挑战2：Reward/Annotation 设计**
- VLM 对复杂时序指令的判断精度如何保证？
- 专门的 temporal semantic reward model 是否需要？
- 如何区分"动作顺序对但过渡不自然"vs"动作顺序错误"？

**挑战3：训练信号精度**
- 如果只做视频级别的时序语义 reward，信号太稀疏
- 需要段级别（segment-level）的时序语义 annotation → 与 DenseDPO 的时序段设计结合

### 与已有工作的创新性组合

**可以创新性组合的技术**：

```
DenseDPO（时序段切分）
    +
VLM 时序语义判断（GPT-4V / VideoLLM：动作时序是否正确？）
    +
Flow-GRPO / DanceGRPO 的在线 RL 框架
    = 时序语义细粒度对齐（本课题）
```

**组合的创新性**：
- 不是 A → intermediate → B 的简单拼接
- DenseDPO 的段切分原本服务于「运动质量」，现在用于「语义动作分段」——两者的偏好对构建方式完全不同（DenseDPO 用 corrupted GT，本课题需要语义重排构建）
- VLM 在 DenseDPO 中仅做运动质量标注，本课题做语义顺序判断——信号维度完全不同

### 选题决策：**推荐，情况3，适合执行**

✅ **Failure case 清晰，无直接竞争者，技术挑战真实但可攻克（有 DenseDPO 模板 + VLM 工具可用）**  
**Novelty 类别：2类（novel pipeline）**  
核心贡献：首个专门针对时序语义指令遵循的视频生成后训练框架

---

## Gap D 分析：去噪轨迹 × 视频帧 二维 Credit Assignment

### Failure Case（精确化）

**问题**：在 GRPO 训练中，所有去噪步骤均等地更新参数，但实际上：
- **早期去噪步骤（高噪声 t≈T）**：决定视频的全局结构（动作类型、主体位置）
- **晚期去噪步骤（低噪声 t≈0）**：决定视觉细节（纹理、色彩、边缘）

**失败模式**：
- 模型通过优化容易的晚期步骤来提高 reward（reward hacking on easy steps）
- 早期去噪步骤的结构性错误（动作不正确）未被有效纠正
- 同一 reward 信号在帧维度也是均等的，但某些关键帧（动作转折点）比背景帧更重要

### Well-established Solution 判断

| 领域 | 是否有类似解决方案 | 说明 |
|---|---|---|
| T2I 去噪步级 credit | ✅ | Step-level Reward for Free (2505.19196) 用 cosine 相似度量化每步贡献 |
| Video latent process reward | ✅ | PRFL (2511.21541) 在 latent 空间全链路 reward（但帧维度均等） |
| Consistent Noisy Latent Rewards | ✅ | 轨迹偏好优化（本方向已有开拓性工作） |
| Video 帧级别 reward | ⚠️ 部分 | DenseDPO 做了时序段的视觉 reward，但不在去噪轨迹维度 |

**判断：情况2** — T2I 去噪步级已有完善方案；Video 的「去噪步骤 × 视频帧」二维联合 credit 是扩展，但技术路径过于清晰（T2I 方案 + DenseDPO 时序段 → 结合即可）。

**选题决策：不推荐作为主课题**（情况2）

技术挑战真实，但属于对已有方案的组合扩展，novelty 较弱（最多 3-4 类）。  
可以作为更大方向中的一个技术模块，而不是主要贡献。

---

## Gap E 分析：通用跨域视频 Reward Model

### Failure Case（精确化）

当前 reward model 格局：
- 物理专用：PhysicsRM, SoliReward（物理域）
- 身份专用：Identity reward model（人体一致性）
- 运动专用：RealDPO（运动真实性）
- **没有**覆盖所有维度的通用视频 reward model

**失败模式**：训练一个视频生成器需要同时优化物理+身份+运动+语义，但目前只能单独优化各维度，多目标组合时各 reward 相互干扰。

### Well-established Solution 判断

| 领域 | 是否有类似解决方案 | 说明 |
|---|---|---|
| 图像通用 reward | ✅ | ImageReward, PickScore, HPS-v2 都是通用图像 reward model |
| Video 质量评测 | ✅ 部分 | VBench/EvalCrafter 多维度，但不是 RL reward 格式 |
| VLM as reward | ✅ | Diffusion-DRF 用冻结 VLM，但不是专用 reward model |

**判断：情况2** — 图像域的通用 reward model 路线已经成熟，视频域是明确的迁移问题，技术路径清晰。

**选题决策：不推荐**（情况2）

---

## Gap F 分析（新识别）：复杂空间+时序组合指令遵循

### Failure Case（精确化）

**Case F1：空间+时序双重组合**
- Prompt：*"A red car is on the left lane, while a blue bus overtakes it from the right. Then the bus turns right at the intersection."*
- 失败模式：空间关系错误（bus 在 left），时序错误（转弯先于超车），或两者都错

**Case F2：多主体并发空间动态**
- Prompt：*"While person A on the left sits down, person B on the right stands up simultaneously."*
- 失败模式：A 和 B 动作同步但空间位置互换，或顺序发生而非并发

### Well-established Solution 判断

| 领域 | 是否有类似解决方案 | 说明 |
|---|---|---|
| 图像空间组合生成 | ✅ 部分 | Attend-and-Excite, T2I-CompBench（空间组合） |
| 视频时序语义 | ❌ | 同 Gap C，视频后训练中无时序语义对齐 |
| 空间+时序联合 | ❌ | 无论哪个领域都没有同时处理空间和时序约束的生成后训练 |

**判断：情况3** — 图像域有空间组合（部分方案），视频时序语义无对应，空间+时序联合无任何领域解法。

**与 Gap C 的关系**：Gap F 是 Gap C 的扩展（增加了空间维度）。Gap C 只要求时序顺序正确，Gap F 还要求空间位置正确且空间+时序约束联合满足。

**选题决策：可以做，比 Gap C 更难但更高影响力**

若要做，建议先解决 Gap C（纯时序语义），再扩展到 Gap F（空间+时序）。Gap F 可以是 Gap C 项目的第二阶段。

---

## 最终选题决策

### 汇总表

| Gap | 判断情况 | Novelty | 难度 | 决策 |
|---|---|---|---|---|
| B：多步骤因果链物理 | 情况4 | 极高 | 极难（reward设计无解法） | ⚠️ 高手专项 |
| **C：时序语义细粒度对齐** | **情况3** | **高** | **中等** | **✅ 首选** |
| D：二维 Credit Assignment | 情况2 | 中 | 中 | ❌ 不独立推荐 |
| E：通用 reward model | 情况2 | 低 | 中 | ❌ 不推荐 |
| F：空间+时序组合指令 | 情况3 | 很高 | 中高 | ✅ Gap C 的扩展 |

---

### ✅ 推荐研究课题：时序语义细粒度对齐（Temporal Semantic Fine-grained Alignment）

**研究问题**：
如何使文本到视频生成模型在后训练阶段，更准确地遵循 prompt 中的时序语义结构（动作顺序、并发关系、时间边界）？

**核心 Failure Case**：
- 现有模型（含 VideoDPO/DenseDPO 后训练）生成的视频忽视 prompt 中"first...then...finally..."或"while A, B"等时序语义结构，产生动作顺序错误、动作遗漏、或时序边界不清的视频。

**技术挑战（真实的）**：
1. **偏好对构建**：如何高效构建「时序语义正确 vs 时序语义错误」的视频偏好对？
2. **Reward 设计**：如何用 VLM 或专用模型评估段级别的时序语义正确性？
3. **训练目标**：需要段级别的时序语义信号（不能只有 video-level reward），如何与 DPO/GRPO 框架结合？

**Novelty 类别**：**2类 novelty（novel pipeline）**

**与已有技术的创新组合**：
```
时序段切分（来自 DenseDPO）
    + 语义动作识别/VLM 时序判断（来自视频理解领域）
    + DPO/Flow-GRPO 框架（来自 VideoDPO/Flow-GRPO）
    ↓
Temporal Semantic DPO（TempoSemanticDPO 或类似命名）
```

**差异化论证**（为什么不是简单组合）：
- DenseDPO 的段切分是为「视觉运动质量」设计的（corrupted GT pairs），本工作需要为「语义顺序正确性」设计全新的 pair 构建策略
- VLM 在 DenseDPO 中做运动 artifact 判断，本工作做「动作时序语义判断」——信号性质完全不同
- 两者组合不能直接拼接（组合需要创新：如何用时序打乱/语义替换来构建负样本，而不是 corrupted GT）

---

### 备选课题：Gap B（多步骤因果链物理）

若有以下条件，可以选 Gap B：
1. 有物理仿真工具（如 Isaac Sim）可以生成因果链的 ground truth
2. 有 VLM/专用模型评估因果连接性的工具（或愿意自己训练）
3. 接受高风险高回报的研究路径

---

## 下一步：解题阶段

选定 **Gap C（时序语义细粒度对齐）** 后，下一步按方法论进入解题阶段：

1. **列出所有可能的 pipeline 方案**（challenge-insight tree 中相关技术的组合）
2. **逐一分析各 pipeline 的优劣势**
3. **选择一个 pipeline** 进行深入设计
4. **设计实验验证计划**：baseline 模型、评测指标（如何评测时序语义对齐？需要新 benchmark）

---

*文档创建时间：基于 Literature Tree 和 Challenge-Insight Tree 的分析*
