### FPN 空间建模改造方案（面向 reg3 + ndvi + ratio，多层特征 & separate_bottlenecks）

> 目标：在**保持 DINOv3 + LoRA 框架不变**的前提下，引入显式空间建模（FPN neck），让模型“同时、统一地”利用整套 patch tokens 来训练与预测：
> - **reg3**：主回归（当前配置常为 Dry_Total_g，训练在 g/m² 的 z-score/可选 log1p 空间）
> - **ndvi**：标量 NDVI（Pre_GSHH_NDVI；可选扩展为 dense NDVI）
> - **ratio**：三类比例（Dry_Clover/Dry_Dead/Dry_Green 的 softmax 比例）
> 并充分利用 `model.backbone_layers.*` 的**多层特征**与 `separate_bottlenecks` 的“每层独立投影/瓶颈”思想。

---

### 1) 现状与痛点（为什么要 FPN）

当前 `use_patch_reg3=true` 的主 reg3 路径本质是：对每个 patch token **独立**做一次回归，再对所有 patch 预测**均值**聚合为图像级输出。

- **缺少 token-to-token 的显式交互**：
  \(\hat y \approx \frac{1}{N}\sum_i f(t_i)\)。即便 MLP 变大，它仍很难表达“覆盖率/纹理/空间结构/混合区域”等与产量强相关的模式。
- **ratio / 5D 仍高度依赖全局 mean(patch)**：即使 reg3 走 patch-mode，ratio 仍主要从全局向量预测，空间信息利用有限。
- **非正方形输入 reshape 风险**：项目里现有的 token→2D map 推断在部分路径用 `sqrt(N)`（对 480×960 的 30×60=1800 patch 网格并不安全），而 FPN 强依赖正确的 (Hp,Wp)。

---

### 2) 设计原则与约束

- **不变**：DINOv3 主干、LoRA 注入方式、冻结策略（`freeze_backbone=true` + LoRA 可训练）、以及“head 轻量导出 / 推理时 backbone+head 两输入”的整体框架。
- **可变**：head/neck 的结构、训练时的特征流、multi-layer 的用法、以及推理脚本中 head 的构建与调用方式。
- **保持训练语义一致**：
  - `y_reg3` 在 dataset 中已是（g/m² → 可选 log1p → 可选 zscore）后的监督信号
  - `y_ndvi` 可选 zscore
  - ratio/5D 的构造与损失尽量与现有实现一致，以便横向对比。

---

### 3) 总体架构草图

#### 3.1 输入：来自 DINOv3 的多层 patch tokens

- 若 `model.backbone_layers.enabled=true`：通过 `feature_extractor.forward_layers_cls_and_tokens(images, indices)` 获取：
  - `pt_list`: list[(B, N, C)]  —— 各层 patch tokens
  - `cls_list`: list[(B, C)] —— 各层 CLS token（可选使用）
- 若关闭多层：退化为使用最后一层 `forward_cls_and_tokens` 的 tokens。

**关键要求：可靠地恢复 patch 网格 (Hp,Wp)**
- 推荐：从输入图像尺寸 (H,W) 与 ViT patch_size（DINOv3 默认 16）计算
  - `Hp = H // patch_size`, `Wp = W // patch_size`
  - 对非整除情况做显式处理（resize 保证整除 / 或用 padding/插值策略）。
- 避免 `sqrt(N)` 推断。

将 `pt` reshape 为 feature map：
- `pt_map = pt.transpose(1,2).reshape(B, C, Hp, Wp)`

> 注：DINO tokens 已内含位置编码，但在卷积/金字塔里仍可考虑拼接 2D 坐标通道（CoordConv）增强几何感知（可选）。

#### 3.2 多层融合 + separate_bottlenecks 对应关系

把 `separate_bottlenecks` 语义迁移为 **每层独立 lateral projection**：
- 若 `use_separate_bottlenecks=true`：每个 backbone layer 具有独立投影 `proj_l`（例如 1×1 Conv + Norm）
- 若 `false`：所有层共享同一个投影（参数更少，可能泛化更稳）

投影到统一通道数：
- `C (DINO embed_dim) -> fpn_dim`（建议 256/384）

然后做“深度维”（layer 维）融合，给 FPN 产生若干个“类 CNN backbone 的 stage 特征” `C2/C3/C4...`。

---

### 4) 推荐的 FPN 具体方案（适配 ViT：同分辨率 tokens → 多尺度金字塔）

#### 4.1 方案核心：把 layer groups 映射为 FPN 的不同尺度输入

DINO 不同 layer 的 tokens **分辨率相同**，但语义层次不同。要得到多尺度，我们引入“受控下采样”构造 C3/C4 等。

- 设选择的层索引为 `L` 个（已排序）。将其分组为 `G` 组（建议 3 或 4 组）：
  - `group_0`（偏浅层）→ 负责高分辨率语义：生成 `C2`（Hp×Wp）
  - `group_1`（中层）→ 生成 `C3`（约 Hp/2×Wp/2）
  - `group_2`（深层）→ 生成 `C4`（约 Hp/4×Wp/4）
  - 可选 `group_3` → `C5`

**组内融合（同尺度）**：
- 对组内每层：`pt_map_l -> proj_l(pt_map_l) -> fpn_map_l`
- 组内聚合：
  - 简单平均 / learnable scalar 权重（softmax）/ 逐像素 layer-attention（三者任选其一）

**跨尺度构造（下采样）**：
- 从组内融合得到的 `C2_full`（Hp×Wp）继续用 stride-2 conv 或 pooling 构造更低分辨率：
  - 但更推荐：对 `group_1/group_2` 的融合结果分别做下采样到对应尺度（避免所有尺度都从同一张图衍生）。
- 由于 Hp/Wp 可能是奇数（如 Hp=30, Wp=60），下采样时统一使用：
  - `stride=2` conv + `padding=1`（或）
  - `F.interpolate` 到目标大小，再 3×3 conv refine

#### 4.2 标准 FPN top-down 融合

得到 `C2, C3, C4(, C5)` 后，构造：
- `P4 = conv1x1(C4)`
- `P3 = conv1x1(C3) + upsample(P4)`
- `P2 = conv1x1(C2) + upsample(P3)`

并对各 `P*` 做 3×3 conv smooth。

> 这样我们得到一个真正的空间金字塔：既包含“深层语义”，又保留“更细分辨率的空间结构”。

---

### 5) 任务头设计：reg3 / ndvi / ratio 三任务共享 FPN

#### 5.1 多尺度全局向量（供 reg3 / ratio / ndvi 标量）

从 `P2/P3/P4` 做全局池化并拼接：
- `g2 = GAP(P2)`, `g3 = GAP(P3)`, `g4 = GAP(P4)`
- `g = concat([g2,g3,g4])`（必要时加 LayerNorm）

然后分别接轻量 MLP/Linear：
- **reg3 head**：`g -> reg3_logits (B, num_outputs_main)`
- **ratio head**：`g -> ratio_logits (B, 3)`（softmax 后与 `y_ratio` 做 MSE）
- **ndvi head**：`g -> ndvi_pred (B,1)`

> 这一步已经比“单 token/均值 token”强：因为 g 来自卷积/金字塔融合过的空间表征。

#### 5.2 可选：空间密集头（让 FPN 的优势更直接）

- **Dense NDVI（推荐可选项）**：
  - `P2 -> conv tower -> ndvi_map_patch (B,1,Hp,Wp)`
  - 将 NDVI dense PNG 标签下采样到 patch 网格，对有效像素做 mask MSE。
  - 这能为 FPN 提供强空间监督，常见地提升泛化。

- **reg3 density map（可选进阶）**：
  - `P2 -> conv tower -> reg3_map_patch (B,D,Hp,Wp)`
  - 通过 `mean/sum` 聚合到图像级与 `y_reg3` 对齐。
  - 需额外正则（平滑/稀疏/非负）以缓解弱监督下的退化解（可先不做）。

---

### 6) 训练与损失：与现有实现对齐

#### 6.1 reg3
- 监督：`y_reg3`（已在 dataset 端完成 g→g/m²、log1p、zscore）
- 预测：`pred_reg3_logits`（同空间域）
- loss：masked MSE（保持现有 `reg3_mask` 语义；CSIRO 通常全 1）

#### 6.2 ratio + 5D
- ratio 预测：`ratio_logits -> softmax -> p_pred`
- ratio loss：对 `p_pred` 与 `y_ratio` 的 MSE（保持当前实现）
- 5D loss：用 `pred_total_gm2` × `p_pred` 构造 5D（并加权 MSE），保持现有逻辑与权重。

#### 6.3 ndvi（标量）
- 预测：`ndvi_pred`
- loss：masked MSE（保持 `ndvi_mask`）

#### 6.4 不确定性加权（UW）与任务集合
- 保持 `loss.weighting: uw` 时的任务拆分：`reg3 / ratio / biomass_5d / ndvi`（以及其它可选任务）。

#### 6.5 数据增强与 mixup/cutmix 的兼容
- CutMix：仍在 image 级别做（不变）。
- Manifold mixup：
  - 推荐继续在 DINO patch tokens 上做（类似当前 patch-mode mixup），然后再进 FPN；
  - 或者在 `C2/C3/C4` 上做（更贴近空间任务，但需要更严谨的对齐）。

---

### 7) multi-layer 与 separate_bottlenecks 的落地规则（建议写进代码的“约定”）

- `model.backbone_layers.enabled=false`：
  - 只用最后层 tokens → 仅构造 `C2`，其余尺度由 neck 自行下采样生成（仍可形成 P2/P3/P4）。

- `enabled=true`：
  - 用 `indices` 取多层 tokens；
  - 先按分组规则映射到 `C2/C3/C4`（见 4.1）；
  - `separate_bottlenecks=true`：每个 layer 有独立 `proj_l`（以及可选独立 norm）；
  - `false`：共享 `proj`（但仍可保留 layer-wise scalar 权重）。

- 分组策略建议：
  - 固定 3 组：按索引均分（便于配置与复现）
  - 或配置化：`fpn.layer_groups: [[...],[...],[...]]`

---

### 8) 需要同步改动的工程面（为了训练 + 导出 + 推理闭环）

> 下面是“计划”，不是立即改代码。

#### 8.1 新 head/neck 的模块边界
- 建议新增：
  - `src/models/spatial_neck_fpn.py`（或同名文件）：实现 Token→Map、layer grouping、FPN、以及多任务 heads。
  - `BiomassRegressor` 内把当前 `shared_bottleneck + reg3_heads + ratio_head + ndvi_head` 重构为：
    - `self.spatial_neck`（FPN）
    - `self.task_heads`（reg3/ratio/ndvi 等）

#### 8.2 HeadCheckpoint 导出
当前 `HeadCheckpoint` 能导出 MLP/多层 bottleneck 的轻量权重。引入 FPN 后需要：
- `meta.head_type: "fpn"`
- 保存必要超参：`fpn_dim / num_levels / layer_indices / layer_groups / use_separate_bottlenecks / use_cls_context / patch_size` 等
- `state_dict`: 保存 neck+heads 的全部参数（仍然很轻）
- 推理侧根据 `head_type` 实例化对应 head module。

#### 8.3 infer_and_submit_pt.py 推理
- patch-mode 推理应改为：
  - backbone 前向得到（多层）tokens → reshape map → head.forward(images/tokens) 输出 `preds_main, preds_ratio, preds_ndvi`
- 保持现有“按 head 的 z_score.json 做反归一化”和“g/m² → grams”的流程不变。

---

### 9) 分阶段落地路线（强烈建议）

- **Phase A（最小可用，尽快验证提升）**
  - FPN 只做 `P2/P3/P4` + 多尺度 GAP，输出 reg3/ratio/ndvi 标量
  - 不做 dense head、不做 density reg3

- **Phase B（引入空间监督，让 FPN 真正学会“哪里是草”）**
  - 加入 dense NDVI head（使用 NDVI dense 数据集）
  - 同时保留 ndvi 标量 head（两者可一起训练或择一）

- **Phase C（进阶）**
  - 尝试 reg3 density map + 聚合约束（配套正则）
  - 或加入 per-patch ratio logits（再聚合为全局比例）

---

### 10) 配置建议（以现有 `configs/train.yaml` 为基准）

新增（示例命名，最终以你们习惯为准）：
- `model.head.type: fpn`
- `model.head.fpn_dim: 256`
- `model.head.num_levels: 3`（P2/P3/P4）
- `model.head.layer_grouping: auto | explicit`
- `model.head.use_cls_context: false/true`（可选：把 CLS 作为全局 gating）
- `model.head.ndvi_dense: true/false`（可选）

保持：
- `model.backbone_layers.enabled / indices / separate_bottlenecks`
- `loss.weighting: uw`

---

### 11) 风险点与对策

- **Hp/Wp 推断错误会直接毁掉空间 head**：必须从 (H,W,patch_size) 得到正确网格。
- **ViT tokens 的空间对齐**：不同数据集可能不同 resize；需保证训练与推理一致的 resize。
- **多层分组选择**：建议先用“按 index 均分”的 deterministic 策略，后续再调优。
- **弱监督的 density reg3**：建议放到 Phase C，先用全局回归验证 FPN 框架价值。

---

### 12) 预期收益（我们希望看到什么）

- reg3：对“空间异质性”更敏感（覆盖率/裸地/阴影/条带），val_r2 更稳。
- ratio：从更强的空间表征得到更合理的比例估计，5D weighted MSE 降。
- ndvi：若引入 dense 监督，通常会显著提升空间表征质量，间接提升 biomass。

---

如果你们认可这个方案，我下一步可以把它拆成**具体的改动清单（按文件/类/函数）**与“最小可跑通的 Phase A”实施路线（仍然先不直接写代码，先把接口/张量形状/导出 meta 字段定死，避免后续推理脚本返工）。
