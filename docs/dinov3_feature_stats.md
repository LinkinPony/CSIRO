### `analyze_dinov3_features.py` 脚本说明

这个脚本用于在**训练数据集**上分析 DINOv3 骨干网络的中间特征，主要聚焦于：

- **每一层（Transformer block）输出的 CLS token 与 patch tokens（pre-norm）**
- **同一层在两种不同归一化方式下的输出：**
  - 使用该层的**独立 LayerNorm**（新建的 `nn.LayerNorm(affine=False)`，称为 `per_layer_ln`）
  - 使用 DINOv3 模型内部共享的**全局输出 LayerNorm**（`self.norm` / `cls_norm`，称为 `global_ln`）

脚本不会保存所有样本的完整特征，而是**对每个层、每种表示、每种 token 角色（CLS / patch）做通道级统计聚合**，输出为一个 `.pt` 文件，供后续渲染与可视化。

---

### 一、运行方式

在仓库根目录下运行：

```bash
python analyze_dinov3_features.py \
  --project-dir . \
  --config configs/train.yaml \
  --dino-weights-pt dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pt \
  --head-weights weights/head \
  --max-images 0 \
  --device auto
```

- **`--project-dir`**: 项目根目录（包含 `configs/` 与 `src/`），默认 `.`。
- **`--config`**: 训练配置文件路径，默认 `project-dir/configs/train.yaml`。
- **`--output`**: 统计结果输出路径。若**不指定**：
  - 从 `train.yaml` 中读取 `version` 和 `logging.log_dir`，构造基础路径：
    - `base_log_dir = logging.log_dir`（默认 `outputs`）
    - `log_dir = base_log_dir/version`（若 `version` 非空），否则 `base_log_dir`
  - 若 `train_all.enabled: true`，优先使用 `train_all` 子目录：
    - `analysis_base_dir = log_dir/train_all`
  - 否则：
    - `analysis_base_dir = log_dir`
  - 默认结果文件路径为：
    - `analysis_base_dir / "feature_stats" / "dinov3_feature_stats.pt"`
    - 例如：
      - 有 `version` + `train_all.enabled: true` 时：`outputs/<version>/train_all/feature_stats/dinov3_feature_stats.pt`
      - 只有 `version` 时：`outputs/<version>/feature_stats/dinov3_feature_stats.pt`
      - 没有 `version` 时：`outputs/feature_stats/dinov3_feature_stats.pt`
- **`--dino-weights-pt`**: 冻结的 DINOv3 骨干 `.pt` 权重文件路径；若省略，则复用 `infer_and_submit_pt.py` 中的默认路径。
- **`--head-weights`**: 回归 head 权重文件或目录，用于从 head 的 meta 中恢复 LoRA 配置；若省略，则复用 `infer_and_submit_pt.py` 中的默认路径。
- **`--max-images`**: 限制参与统计的训练图像数（0 表示使用整个训练集）。
- **`--device`**: 计算设备，`auto` / `cuda` / `cpu`，默认 `auto`。

脚本内部会：

- 使用 `train.yaml` 中的数据配置，通过 `PastureDataModule` 构建训练集 dataloader（与 `train.py` 一致）。
- 使用与 `infer_and_submit_pt.py` 一致的逻辑构建 DINOv3 骨干，并从 head 权重中恢复 LoRA 配置。
- 在训练集上做一次前向推理，累积各层特征统计，并最终通过 `torch.save` 写入结果文件：
  - 若显式传入 `--output`，则使用该路径；
  - 否则使用上述基于 `version` 与 `train_all` 的默认路径。

---

### 二、输出文件结构

输出文件通过 `torch.save(payload, output_path)` 保存，`payload` 是一个嵌套的 `dict`，主要包含两部分：

```python
payload = {
    "meta": {...},
    "layers": {...},
}
```

#### 1. `meta` 字段

- **`meta["project_dir"]`**: 运行时使用的项目根目录字符串。
- **`meta["config"]`**: 解析后的 `train.yaml` 完整配置（字典）。
- **`meta["backbone"]`**: 与骨干及 LoRA 相关的元信息：
  - `backbone_name`: e.g. `"dinov3_vith16plus"` 或 `"dinov3_vitl16"`。
  - `dino_weights_pt`: 实际加载的 DINOv3 权重文件绝对路径。
  - `head_weights_base`: 用于发现 head 权重的基路径（文件或目录）。
  - `first_head_path`: 用于恢复 LoRA 的第一个 head 权重文件路径。
  - `used_lora`: 是否从 head 的 `peft` payload 成功注入了 LoRA。
  - `lora_error`: 若 LoRA 注入失败，这里会给出错误信息字符串，否则为 `None`。
- **`meta["counts"]`**: 数据规模信息：
  - `num_images`: 实际参与统计的训练图像数量（受 `--max-images` 限制）。
  - `num_batches`: 遍历的 batch 数量。
  - `num_layers`: DINOv3 Transformer block 的层数（例如 32）。
  - `embedding_dim`: 通道维度（例如 1280）。
  - `n_storage_tokens`: 每层的 storage token 数量（一般为 4）。

#### 2. `layers` 字段

`layers` 是按层组织的嵌套字典：

```python
layers: Dict[int, Dict[str, Dict[str, Dict[str, Tensor]]]]
```

结构如下：

- 最外层 key：**`layer_idx`**（int），对应 Transformer block 的索引（0-based）。
- 第二层 key：**`rep_name`**，表示特征类型：
  - **`"pre"`**: 全局 LayerNorm 之前的 block 输出（即 `_get_intermediate_layers_not_chunked` 的结果）：
    - `CLS` 为 `out[:, 0, :]`
    - patch tokens 为 `out[:, 1 + n_storage_tokens :, :]`
  - **`"per_layer_ln"`**: 对该层的 `out` 使用**独立的 `nn.LayerNorm(affine=False)`** 做归一化后的结果。
  - **`"global_ln"`**: 使用 DINOv3 模型内部共享的 `self.norm` / `cls_norm` 做归一化后的结果（即 paper 中的输出端 LayerNorm）。
- 第三层 key：**`role`**，token 角色：
  - `"cls"`: CLS token 通道统计。
  - `"patch"`: 所有 patch tokens（在 batch 与空间维度上展平后）的通道统计。
- 最内层 value：通道级统计字典（均为 CPU 上的 `torch.Tensor`，`float32`）：

```python
layers[layer_idx][rep_name][role] = {
    "count": int,                  # 聚合时的 token 数（CLS: images 数；patch: 所有 patch token 总数）
    "mean": Tensor[C],             # 每个通道的均值
    "std": Tensor[C],              # 每个通道的标准差（数值上 clamp 到 >= 0）
    "max_abs": Tensor[C],          # 每个通道在所有 token 上观测到的最大绝对值
}
```

其中：

- `C = embedding_dim`（例如 1280）。
- 对于 `"patch"` 角色，统计是在 **所有图像 × 所有 patch tokens** 的维度上累积的。

---

### 三、如何解析与可视化

在后续分析脚本或 notebook 中，可以按如下方式加载并使用这些统计数据：

```python
import torch

# 根据你的 config 中的 version / train_all 设置调整路径：
# 例如 version=manifold-mixup-disable 且 train_all.enabled: true 时：
# path = "outputs/manifold-mixup-disable/train_all/feature_stats/dinov3_feature_stats.pt"
path = "outputs/<version>/feature_stats/dinov3_feature_stats.pt"

payload = torch.load(path, map_location="cpu")
meta = payload["meta"]
layers = payload["layers"]

# 示例：获取第 31 层 global_ln 下 CLS token 的通道统计
idx = 31
stats_cls_global = layers[idx]["global_ln"]["cls"]
mean = stats_cls_global["mean"]   # shape: (C,)
std = stats_cls_global["std"]     # shape: (C,)
max_abs = stats_cls_global["max_abs"]  # shape: (C,)

# 示例：绘制第 31 层 patch token pre-norm 的 max_abs 直方图
import matplotlib.pyplot as plt

max_abs_patch_pre = layers[idx]["pre"]["patch"]["max_abs"].numpy()
plt.hist(max_abs_patch_pre, bins=100)
plt.title("Layer 31 patch tokens (pre-norm) max|x_c| distribution")
plt.xlabel("max |x_c| over tokens")
plt.ylabel("channel count")
plt.show()
```

你可以基于这些统计进行：

- 不同层之间的分布对比（如 `std`、`max_abs` 的层间曲线）。
- 比较 `pre` / `per_layer_ln` / `global_ln` 三种表示的差异，评估共享输出 LayerNorm 对中间层的影响。
- 寻找**极端离群通道**（例如按 `max_abs` 排序，观察 top-k 通道）。

如果后续需要增加采样级别的 token 保存（例如随机子样本），可以在现有脚本的统计流程中额外加入采样逻辑，而不会破坏当前文件结构。


