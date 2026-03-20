---
title: Video Gen Post-Training 研究分析
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

# Video Gen Post-Training 研究分析

按照 Goal-driven research 方法论构建 Literature Tree 和 Challenge-Insight Tree，并识别研究空间。

---

## 一、Literature Tree（Novelty Tree）

### General Goal

**让视频生成模型生成符合物理规律、用户偏好、语义准确、时空一致的高质量视频**（Video Generation Post-Training Alignment）

---

### Milestone Tasks（按研究成熟度排列）

#### MT1: 通用视觉质量 + 语义对齐（General Preference Alignment）

1类novelty（seminal work）：**VideoDPO** (2412.14167)  
首次系统性地将DPO应用于视频扩散模型，提出OmniScore综合衡量视觉质量与语义对齐。

归类论文：
- **VideoDPO** (2412.14167) — 1类 novelty，seminal: DPO for T2V，提出OmniScore
- **AI Feedback for Dynamic Objects** (2412.02617) — 1类 novelty，统一RL微调目标推导，VLM binary feedback
- **Diffusion-LPO** (2510.01540) — 2类 novelty，seminal: Listwise DPO (Plackett-Luce) for diffusion models

---

#### MT2: 物理一致性视频生成（Physics-Consistent Video Generation）

1类novelty（seminal work）：**AI Feedback for Dynamic Objects** (2412.02617) / **PhysMaster** (2510.13809)  
率先探索物理一致性的后训练方法。**RigidBench** 是该方向首个系统性评测。

归类论文（按pipeline细分）：

**P2a：基于DPO的物理对齐**
- **PhysCorr** (2511.03997) — 2类 novelty，seminal: 双维度PhysicsRM（intra-object + inter-object）+ PhyDPO
- **PhyGDPO** (2512.24551) — 2类 novelty，seminal: 物理感知Groupwise DPO（Plackett-Luce）+ LoRA-SR
- **RealDPO** (2510.14955) — 3类 novelty，在DPO下用真实视频作为正样本

**P2b：基于显式物理表示学习**
- **PhysMaster** (2510.13809) — 2类 novelty，seminal: PhysEncoder（物理表示编码器）+ DPO/RL端到端优化
- **ProPhy** (2512.05564) — 3类 novelty，Mixture-of-Physics-Experts (MoPE) + VLM物理对齐迁移

**P2c：多模态条件注入**
- **UnityVideo** (2512.07831) — 2类 novelty，统一多模态多任务SFT（分割/骨架/光流/深度图）
- **ProPhy** (2512.05564) — 3类 novelty（在P2b已列）

**评测**：
- **RigidBench** — 1类 novelty（首个rigid-body physics评测基准）

---

#### MT3: 细粒度时序/空间偏好优化（Fine-grained Temporal/Spatial Preference Optimization）

1类novelty（seminal work）：**DenseDPO** (2506.03517)  
首次在视频DPO中引入时序段级别的精细偏好标注，突破全局偏好比较的局限。

归类论文：
- **DenseDPO** (2506.03517) — 1类 novelty，seminal: corrupted GT pairs + 时序段级别标注 + VLM自动标注
- **LocalDPO** (2601.04068) — 2类 novelty，seminal: 时空区域级别local preference（空间维度扩展）

---

#### MT4: 特定主体身份保持视频生成（Identity/Subject-Preserving Video Generation）

1类novelty（seminal work）：**Identity-GRPO** (2510.14256) / **ID-Crafter** (2511.00511)  
首次将RL/GRPO用于多人/多主体身份一致性视频生成的后训练。

归类论文：
- **Identity-GRPO** (2510.14256) — 1类 novelty，seminal: GRPO for multi-human identity in video diffusion
- **ID-Crafter** (2511.00511) — 2类 novelty，hierarchical identity attention + VLM语义指导 + online RL

---

#### MT5: 长时程时空一致性（Long-horizon Spatial/Temporal Consistency）

1类novelty（seminal work）：**WorldPack** (2512.02473)  
首次通过压缩记忆机制解决长时程视频世界模型的空间一致性问题。

归类论文：
- **WorldPack** (2512.02473) — 1类 novelty，compressed memory（trajectory packing + retrieval）

---

### Pipeline/Representation（2类 novelty）汇总

| Pipeline 类型 | Seminal Work | 核心创新 |
|---|---|---|
| **P1: Offline DPO for Video Diffusion** | VideoDPO (2412.14167) | OmniScore; 自动preference pair构建 |
| **P2a: Online RL (GRPO) for Flow Matching** | Flow-GRPO (2505.05470) | 首次将在线RL引入flow matching；ODE→SDE转换 |
| **P2b: Unified GRPO for Diffusion+Flow** | DanceGRPO (2505.07818) | 统一GRPO适配diffusion+rectified flow；T2I+T2V+I2V |
| **P2c: Task-specific GRPO for Video** | Identity-GRPO (2510.14256) | GRPO variant for video; 身份保持reward model |
| **P3: Listwise/Groupwise DPO** | Diffusion-LPO (2510.01540) | Plackett-Luce ranked list for diffusion |
| **P4: Real-data as Positive Pipeline** | RealDPO (2510.14955) | 真实视频vs.生成视频; RealAction-5K |
| **P5: Dense/Local Preference Pipeline** | DenseDPO (2506.03517) | corrupted GT pairs; 时序段标注 |
| **P6: Free Differentiable Reward** | Diffusion-DRF (2601.04153) | 冻结VLM为critic; 无需reward model训练 |
| **P7: Latent Process Reward Pipeline** | PRFL (2511.21541) | 全去噪链latent空间reward; 无需VAE decoding |
| **P8: Multi-modal World Conditioning** | UnityVideo (2512.07831) | 动态noising统一多模态训练范式 |
| **P9: Geometry Prior DPO** | VideoGPA (2601.23286) | 几何基础模型提供dense preference signal |

---

### Module-level Innovations（3/4类 novelty）汇总

| 模块 | 来源论文 | 解决问题 |
|---|---|---|
| **OmniScore** | VideoDPO | 多维preference打分 (质量+语义) |
| **PhysEncoder** | PhysMaster | 物理表示编码+DPO端到端 |
| **PhysicsRM** | PhysCorr | 双维度物理reward model |
| **MoPE (Mixture-of-Physics-Experts)** | ProPhy | 分层物理先验提取 |
| **MimicryDiscovery Cycle (MDcycle)** | PhysRVG | 严格执行rigid body物理规则的RL范式 |
| **LoRA-SR (LoRA-Switch Reference)** | PhyGDPO | 高效DPO reference避免全模型复制 |
| **Region-aware DPO loss** | LocalDPO | 限定corrupted区域的局部偏好学习 |
| **Cross-prompt pairing** | SoliReward | 避免annotation noise |
| **Hierarchical Progressive Query Attention** | SoliReward | reward model特征聚合 |
| **Dense VQA Decomposition** | Diffusion-DRF | prompt分解为多维QA，信息丰富反馈 |
| **Hierarchical Identity Attention** | ID-Crafter | 多层次主体特征聚合 (intra/inter/cross-modal) |
| **Trajectory Packing + Memory Retrieval** | WorldPack | 长时程压缩记忆机制 |
| **ODE-to-SDE Conversion** | Flow-GRPO | 确定性ODE转随机SDE以支持RL采样 |
| **TaRoS (Target-Robust Reward Signaling)** | TaRoS (2511.19356) | 组内稀疏性+饱和度自适应下调，防止reward hacking |
| **PRFL (Process Reward Feedback Learning)** | PRFL (2511.21541) | latent空间全链路reward，无VAE解码开销 |
| **Cosine-similarity Step Credit** | Step-level Reward (2505.19196) | 通过cosine相似度变化量化每步去噪贡献 |
| **Geometry Foundation Model DPO** | VideoGPA | 用geometry模型自动提取dense 3D一致性signal |

---

## 二、Challenge-Insight Tree

### C1: 生成视频违反物理定律（Physics Inconsistency）

**Challenge 细分**：
- 运动学违反（物体穿墙、违反重力、速度不合理）
- 动力学违反（碰撞后反应错误、刚体形变不当）
- 多物体交互错误（相互作用不符合力学原理）

**Insights（已有解决方案）**：
- 用真实视频作为正样本，让模型从真实物理现象中学习 → **RealDPO**
- 用VLM的物理推理能力生成物理reward进行DPO优化 → **PhyGDPO**, **PhysCorr**
- 训练专用物理表示编码器，将物理知识注入视频生成条件 → **PhysMaster**
- 用Mixture-of-Experts分层提取语义级+token级物理先验 → **ProPhy**
- 建立双维度reward（intra-object稳定性 + inter-object交互） → **PhysCorr (PhysicsRM)**
- 用physics simulation数据构建大规模训练集（PhyAugPipe, PhyVidGen-135K） → **PhyGDPO**
- 注入多模态物理条件（光流、深度图、骨架）辅助世界感知 → **UnityVideo**

**尚未解决的子问题**：
- **复杂因果链物理**：多步骤时序因果（A触发B触发C）的物理一致性，无人解决
- **3D几何一致性**：透视关系正确性、深度一致性、遮挡关系，未被后训练方法覆盖

---

### C2: 偏好标注质量差 / 效率低（Annotation Quality & Efficiency）

**Challenge 细分**：
- Pairwise全局比较导致ambiguous supervision signal
- 标注者偏向低运动视频（低动态视频 artifact 少，导致motion bias）
- 人工标注成本高，难以规模化

**Insights（已有解决方案）**：
- 用corrupted GT视频对代替独立noise采样对，消除motion bias → **DenseDPO**
- 时序段级别细粒度标注，信号更精确 → **DenseDPO**
- 时空区域级别局部标注 → **LocalDPO**
- VLM自动标注（GPT segment-level preference）→ **DenseDPO**
- Listwise/Groupwise ranking（超越pairwise） → **Diffusion-LPO**, **PhyGDPO**
- 单item二元标注 + cross-prompt配对 → **SoliReward**
- 真实视频自动充当正样本，无需人工pair标注 → **RealDPO**, **PhyGDPO**

**尚未解决的子问题**：
- **时序语义对齐标注**：针对复杂时序指令（"先做A，再做B"）的细粒度语义对齐，无对应标注方案

---

### C3: Reward Hacking & Reward Model噪声（Reward Design）

**Challenge 细分**：
- 学习型reward model容易被模型exploit（reward hacking）
- Reward model基于in-prompt pairwise标注，噪声大
- 单一scalar reward粒度粗，信用分配困难

**Insights（已有解决方案）**：
- Cross-prompt pairing避免in-prompt annotation噪声 → **SoliReward**
- Modified BT loss处理win-tie场景，规范化score分布 → **SoliReward**
- 用冻结VLM作为free critic，不训练reward model → **Diffusion-DRF**
- 将prompt分解为多维QA，提供dense信息反馈 → **Diffusion-DRF**
- 双维度专用reward（物理域） → **PhysCorr**
- 去噪轨迹级别的dense reward → **Consistent Noisy Latent Rewards**

**尚未解决的子问题**：
- **跨domain通用reward model**：能泛化到物理/身份/运动/语义等多维度的统一reward
- **Trajectory reward的最优信用分配**：如何对去噪轨迹每一步最优地分配reward credit

---

### C4: 复杂运动生成困难（Complex Motion Generation）

**Challenge 细分**：
- 人体活动的自然流畅动作难以生成
- 物体间动态交互（抛接、碰撞）不自然
- 复杂手势、精细肢体动作失真

**Insights（已有解决方案）**：
- 真实世界动作视频作为正样本（RealAction-5K） → **RealDPO**
- VLM binary feedback专门针对物体动态交互 → **AI Feedback paper**
- 统一RL微调目标分析（KL regularization, policy projection） → **AI Feedback paper**

---

### C5: 多主体/身份一致性（Multi-subject Identity Preservation）

**Challenge 细分**：
- 多人场景中角色外貌随帧变化
- 多个主体之间语义冲突
- 动态交互中身份混淆

**Insights（已有解决方案）**：
- Hierarchical identity attention（intra-subject → inter-subject → cross-modal） → **ID-Crafter**
- GRPO + 大规模人体一致性preference数据集 → **Identity-GRPO**
- Online RL持续精化 → **ID-Crafter**

---

### C6: 长时程一致性（Long-horizon Consistency）

**Challenge 细分**：
- 长视频中空间关系前后矛盾
- 场景语义随时间漂移
- 计算成本随上下文长度爆炸

**Insights（已有解决方案）**：
- 压缩记忆：trajectory packing + memory retrieval → **WorldPack**
- 多模态条件注入（分割、深度）提供持续世界感知约束 → **UnityVideo**

---

## 二-补充、新增论文（Literature Tree 扩充）

以下论文在初版之后被发现，已补充入 Literature Tree：

### MT2 物理一致性补充
- **PhysRVG** (2601.11087) — 2类 novelty，首次在video gen中严格执行rigid body物理碰撞规则（直接在高维空间约束而非作为条件），构建PhysRVGBench评测。使用MDcycle框架。

### P2 Online RL Pipeline 补充
- **Flow-GRPO** (2505.05470) — **1类 novelty（seminal）**，首次将在线policy gradient RL引入flow matching模型；ODE→SDE转换 + Denoising Reduction策略，T2I；Diffusion-DRF比较基线。
- **DanceGRPO** (2505.07818) — **1类 novelty（seminal）**，首个统一GRPO框架适配diffusion + rectified flow两种范式，覆盖T2I+T2V+I2V，同时支持5种reward model。
- **TaRoS** (2511.19356) — 3类 novelty，针对Video GRPO中的Goodhart's Law问题（reward饱和+shortcut），提出Target-Robust Reward Signaling（组内稀疏性+饱和度自适应下调）。

### P7 Trajectory/Process Reward 补充
- **PRFL** (2511.21541) — **2类 novelty（seminal）**，在noisy latent空间建立Process Reward，利用预训练视频生成模型自身作为latent reward model，无需VAE解码，全去噪链优化。
- **Step-level Reward for Free** (2505.19196) — 3类 novelty（T2I），用cosine相似度变化量化每步去噪贡献，无需额外网络；为P7方向的轻量化方案。

### P9 Geometry Prior DPO（新增 Pipeline）
- **VideoGPA** (2601.23286) — **1类 novelty（seminal，新milestone task）**，首次专门针对T2V的3D几何一致性后训练；用geometry基础模型自动提取dense preference signal，self-supervised DPO。**注：此论文覆盖了研究空白A。**

---

## 三、Goal-driven Research：研究空间分析

### Roadmap（从已成熟到未探索）

```
Level 1 (成熟): 通用视觉质量+语义对齐  ← VideoDPO, Diffusion-LPO
Level 2 (活跃): 单对象物理一致性        ← PhysMaster, PhyGDPO, PhysCorr, ProPhy
Level 3 (新兴): 细粒度时空偏好优化      ← DenseDPO, LocalDPO
Level 4 (新兴): 特定能力对齐(身份/运动)  ← Identity-GRPO, RealDPO
Level 5 (早期): 长时程世界一致性        ← WorldPack
Level 6 (空白): 复杂因果链物理 / 3D几何 / 时序语义细粒度 ← 无工作
```

---

### 重要研究空白（Failure Cases 分析）

#### ~~研究空白 A：3D几何一致性对齐~~ （⚠️ 已被 VideoGPA 覆盖）

**VideoGPA** (2601.23286) 已针对此问题提出 geometry foundation model + self-supervised DPO 方案，本空白基本关闭。
但 VideoGPA 仅用了简单geometry模型（深度/法向量），**以下子问题仍开放**：
- 动态相机运动下的多视图一致性（camera trajectory equivariance）
- 遮挡/出现的几何正确性（occlusion boundary consistency）
- 与物理运动联合的3D+Physics一致性
若选择此方向，需在 VideoGPA 基础上找到更细粒度的 failure case 进入。

---

#### 研究空白 B：复杂多步骤因果链物理（★★★ 优先级：高）

**Failure case**：
- 多步骤因果链（推倒骨牌 → 触发绳子 → 拉动杠杆）中间步骤违反物理
- 现有工作只解决单物体单时刻物理 (重力、碰撞)，不解决时序因果链
- RigidBench等评测也只考虑简单单对象物理

**是否有well-established solution？**
- 情况4：目前各领域都没有针对多步骤因果链的post-training方案
- **结论：高难度，适合有经验的研究者，但发表影响力大**

---

#### 研究空白 C：时序语义细粒度对齐（★★★ 优先级：高）

**Failure case**：
- 提示词"先打开门，然后走进去，最后坐下"中，动作顺序可能错乱
- "从左向右移动"的方向性指令不被遵守
- DenseDPO只针对motion quality的细粒度，不针对语义时序

**是否有well-established solution？**
- 情况3-4之间：LLM有time-step语义对齐（chain-of-thought），但video领域无对应工作
- **结论：是真正的open problem，novelty高**

**技术方向**：
- 将时序语义标注（每个时间段对应哪个子动作）引入DPO信号
- 用VLM对视频进行segment-level语义动作识别，构建时序语义preference pairs

---

#### 研究空白 D：去噪轨迹 Credit Assignment（★★ 优先级：中，需细化方向）

**已有工作**：
- **PRFL** (2511.21541) — latent空间process reward，视频生成全链路
- **Step-level Reward for Free** (2505.19196) — cosine相似度步级credit（T2I）
- **Consistent Noisy Latent Rewards** — 轨迹偏好优化（论文列表中已有）

**是否有well-established solution？**
- 情况2-3之间：T2I领域已有多个方案；视频域的时序credit assignment（跨帧维度）仍未被彻底解决
- **结论：需要聚焦更具体的sub-problem才有novelty空间**

**仍开放的具体问题**：
- 如何在「去噪步骤 × 视频帧」二维时空中联合分配credit？（现有方法只做其中一个维度）
- 视频中早期去噪步骤主要影响动作结构，晚期影响细节——如何针对不同维度给予不同step的reward权重？

---

#### 研究空白 E：通用跨域Reward Model（★★ 优先级：中）

**Failure case**：
- 现有reward model各自独立（物理专用、人体专用、运动专用）
- 没有能统一覆盖多维度的通用video reward model
- Diffusion-DRF虽用VLM但需要prompt工程，泛化能力有限

**是否有well-established solution？**
- 情况2：图像领域有ImageReward、PickScore等通用reward，但video领域没有对应的强通用reward model
- **结论：类比图像领域，有明确的迁移路径，但video挑战更大**

---

### 推荐研究方向（更新版）

⚠️ 注意：VideoGPA 覆盖了原空白A，需要调整优先级。

1. **【首选】时序语义细粒度对齐**：open problem最清晰，无直接竞争者，可借鉴DenseDPO时序段设计 + VLM动作识别，产出2类novelty（novel pipeline）
2. **【高影响】多步骤因果链物理**：当前所有物理工作只做单对象单时刻，多步骤因果链完全空白，发表影响力大但难度高
3. **【中风险，需细化】二维时空 Credit Assignment**：结合「去噪步骤 × 视频帧」二维credit，比现有工作更完整，可作为3类novelty
4. **【补充VideoGPA的子问题】动态相机下的3D+Physics联合一致性**：在VideoGPA基础上，聚焦动态运动场景，可找到明确的failure case

---

## 四、论文分类汇总表

| 论文 | Milestone Task | Pipeline | Novelty类别 |
|---|---|---|---|
| VideoDPO (2412.14167) | MT1 通用对齐 | P1 Offline DPO | **1类** |
| AI Feedback Dynamic (2412.02617) | MT1/MT2 | P1 Offline RL/DPO | **1类** |
| PhysMaster (2510.13809) | MT2 物理一致性 | 物理表示+DPO/RL | **2类** |
| PhysCorr (2511.03997) | MT2 物理一致性 | P1+双reward model | **2类** |
| PhyGDPO (2512.24551) | MT2 物理一致性 | P3 Groupwise DPO | **2类** |
| RealDPO (2510.14955) | MT2/MT6 运动 | P4 Real-data positive | **3类** |
| ProPhy (2512.05564) | MT2 物理一致性 | P8 Multi-modal | **3类** |
| UnityVideo (2512.07831) | MT2 物理/世界 | P8 Multi-modal SFT | **2类** |
| RigidBench | MT2 物理一致性 | 评测基准 | **1类** |
| DenseDPO (2506.03517) | MT3 细粒度时序 | P5 Dense DPO | **1类** |
| LocalDPO (2601.04068) | MT3 细粒度空间 | P5 Local DPO | **2类** |
| Identity-GRPO (2510.14256) | MT4 身份保持 | P2 GRPO | **1类** |
| ID-Crafter (2511.00511) | MT4 身份保持 | P2+SFT hybrid | **2类** |
| WorldPack (2512.02473) | MT5 长时一致性 | P7+ memory | **1类** |
| Diffusion-LPO (2510.01540) | MT1 通用对齐 | P3 Listwise DPO | **2类** |
| Diffusion-DRF (2601.04153) | 通用对齐/reward | P6 Free reward | **2类** |
| SoliReward (2512.22170) | 通用reward设计 | Reward Model | **3类** |
| Consistent Noisy Latent Rewards | MT3/reward | P7 Trajectory reward | **2类** |
| Flow-GRPO (2505.05470) | MT1 通用对齐 | P2a Online RL | **1类** |
| DanceGRPO (2505.07818) | MT1 通用对齐 | P2b Unified GRPO | **1类** |
| TaRoS (2511.19356) | 通用对齐/reward | P2c GRPO改进 | **3类** |
| PhysRVG (2601.11087) | MT2 物理一致性 | P2 RL严格约束 | **2类** |
| VideoGPA (2601.23286) | MT6 新: 3D几何一致性 | P9 Geometry Prior DPO | **1类** |
| PRFL (2511.21541) | MT3/reward | P7 Latent Process Reward | **2类** |
| Step-level Reward (2505.19196) | MT3/reward | P7 Step credit (T2I) | **3类** |
