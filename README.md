### Nano Banana Pro 训练数据增强工具（Gemini / OpenRouter）

本仓库提供一个**离线 AIGC 图像增强**工具：对 `data/train.csv` 中的训练图片做“成像域增强”（光照/曝光/白平衡/噪声/压缩等），**保留原始标签不变**，并输出到 `data/nano_banana_pro/train/`。训练时可通过配置开关把这些增强图当作额外样本加入训练集。

该工具支持两种 API 后端：
- **Gemini 原生 API**（默认，Nano Banana Pro：`gemini-3-pro-image-preview`）
- **OpenRouter**（通过 `https://openrouter.ai/api/v1/chat/completions` 统一调用支持 image 输出的模型）

---

### 快速开始（最短流程）

- **1) 准备 API Key**
  - **Gemini（默认）**：
    - 在仓库根目录创建 `gemini.secret` 文件，内容为你的 Gemini API Key（纯文本一行即可）。
  - **OpenRouter（可选）**：
    - 在仓库根目录创建 `openrouter.secret` 文件，内容为你的 OpenRouter API Key（纯文本一行即可）。
    - 并在 `configs/nano_banana_pro_augment.yaml` 里设置 `api.provider: openrouter`。
  - 以上 secret 文件均已加入 `.gitignore`，不会被提交。

- **2) 运行增强脚本（生成增强图片 + manifest）**

```bash
python tools/nano_banana_pro/augment_train.py --config configs/nano_banana_pro_augment.yaml
```

- **3) 训练时接入增强数据（可选）**
  - 在 `configs/train.yaml` 中打开 `data.aigc_aug.enabled: true`，训练会自动读取 `manifest.csv` 并把增强样本加入 `train_df`。

---

### 脚本：`tools/nano_banana_pro/augment_train.py`

该脚本从 `data/train.csv` 聚合得到 `image_id -> image_path`（每个 `image_id` 对应一张图片），然后对每张图按 YAML 里配置的 `augmentations` 逐个调用 Nano Banana Pro（Gemini 3 Pro Image）进行编辑生成。

#### CLI 参数（全部）

- **`--config`**
  - **作用**：指定增强配置文件路径（YAML）。
  - **默认值**：`configs/nano_banana_pro_augment.yaml`
  - **说明**：相对路径以仓库根目录为基准；也支持绝对路径。

- **`--limit`**
  - **作用**：限制最多处理多少张训练图片（用于小规模验证/冒烟测试）。
  - **默认值**：`0`（表示不限制，处理全部）
  - **示例**：

```bash
python tools/nano_banana_pro/augment_train.py --config configs/nano_banana_pro_augment.yaml --limit 50
```

- **`--force`**
  - **作用**：强制重新生成（即使输出文件已存在/manifest 已记录成功）。
  - **默认值**：关闭
  - **说明**：开启后仍会写入 manifest（用于对比不同 prompt/参数时复写产物）。

---

### 配置：`configs/nano_banana_pro_augment.yaml`

该 YAML 控制：
1) 用哪个模型、如何读 API Key、重试/超时等；
2) 从哪里读取训练数据索引（`train.csv`）；
3) 输出目录、manifest 与日志文件；
4) 具体增强类型与 prompt。

#### 顶层字段

- **`api`**：Gemini / OpenRouter 调用相关配置
- **`data`**：训练数据索引与输出位置配置
- **`prompt_prefix`**：每条增强 prompt 前都会加的通用约束（强烈建议保留）
- **`augmentations`**：增强类型列表（每个元素代表一种增强）

#### `api` 字段说明

- **`api.provider`**
  - **作用**：选择 API 后端
  - **可选**：`gemini` | `openrouter`
  - **默认**：`gemini`

- **`api.model`**
  - **作用**：模型名
  - **说明**：
    - `provider=gemini`：Gemini 模型 id（默认 `gemini-3-pro-image-preview`，Nano Banana Pro）
    - `provider=openrouter`：OpenRouter 的模型 id（例如 `google/gemini-2.5-flash-image-preview`）

- **`api.api_key_file`**
  - **作用**：API key 文件路径
  - **默认**：
    - `provider=gemini`：`gemini.secret`
    - `provider=openrouter`：`openrouter.secret`
  - **说明**：相对路径以仓库根目录为基准；也支持绝对路径。

- **（OpenRouter 可选）`api.base_url`**
  - **作用**：OpenRouter API 地址
  - **默认**：`https://openrouter.ai/api/v1/chat/completions`

- **（OpenRouter 可选）`api.http_referer` / `api.x_title` / `api.extra_headers`**
  - **作用**：额外请求头
  - **说明**：
    - `http_referer` 会作为 `HTTP-Referer` header
    - `x_title` 会作为 `X-Title` header
    - `extra_headers` 可传任意 header（map）

- **（OpenRouter 可选）`api.modalities`**
  - **作用**：请求的输出模态（OpenRouter 图像生成要求包含 `"image"`）
  - **默认**：`["image", "text"]`

- **`api.concurrency`**
  - **作用**：并发调用 API 的并行度（线程数）
  - **默认**：`1`
  - **建议**：适当增大可显著加速生成，但请注意 API 的 rate limit 与成本。

- **`api.concurrency_scope`**
  - **作用**：并发的“作用范围”
  - **可选**：`per_image` | `global`
  - **默认**：`per_image`
  - **说明**：
    - `per_image`：一次只对**同一张图片**的多个变体并行（因此并发上限还会被 `augmentations`/`num_images` 限制）
    - `global`：在**全数据集范围**保持最多 `api.concurrency` 个任务同时在飞（当每张图的变体数较少时推荐）

- **`api.timeout_s`**
  - **作用**：单次请求超时时间（秒）
  - **默认**：`120`

- **`api.max_retries`**
  - **作用**：请求失败后的最大重试次数
  - **默认**：`3`

- **`api.retry_backoff_s`**
  - **作用**：重试退避基准秒数（实际会乘以 attempt）
  - **默认**：`5.0`

- **`api.image_config`（可选）**
  - **作用**：可选的图像生成/编辑参数透传
  - **说明**：
    - `provider=gemini`：透传给 Gemini 的 `generationConfig.imageConfig`（字段通常是 `aspectRatio/imageSize`）
    - `provider=openrouter`：透传给 OpenRouter 的 `image_config`（字段通常是 `aspect_ratio/image_size`）
  - **默认**：未设置（让模型尽量匹配输入图的尺寸/长宽比）
  - **示例**（通常不建议一开始就开，先保持与原图一致更安全）：

```yaml
api:
  image_config:
    aspectRatio: "4:3"
    imageSize: "2K"
```

#### `data` 字段说明

- **`data.data_root`**
  - **作用**：数据根目录（训练图片与 `train.csv` 都在其下）
  - **默认**：`data`

- **`data.train_csv`**
  - **作用**：训练 CSV 文件名/路径（相对 `data_root`）
  - **默认**：`train.csv`

- **`data.primary_targets`**
  - **作用**：只对包含这些主监督目标的图片进行增强（避免和训练目标不一致）
  - **默认**：`[Dry_Total_g]`

- **`data.output_dir`**
  - **作用**：增强结果输出目录
  - **默认**：`data/nano_banana_pro/train`

- **`data.output_ext`**
  - **作用**：输出图片扩展名（由模型返回的 bytes 直接写入）
  - **默认**：`png`

- **`data.manifest`**
  - **作用**：manifest 文件名（写在 `output_dir` 下）
  - **默认**：`manifest.csv`

- **`data.log_file`**
  - **作用**：日志文件名（写在 `output_dir` 下）
  - **默认**：`augment.log`

#### `prompt_prefix`（重要）

每条增强 prompt 前都会拼接 `prompt_prefix`。建议用于强约束：
- **只允许“成像域变化”**（光照/曝光/色彩/噪声/压缩/轻微模糊/镜头伪影）
- **禁止“语义变化”**（植被密度、枯绿比例、覆盖边界、构图、几何形变、重绘等）

#### `augmentations` 字段说明

`augmentations` 是一个列表，每个元素是：
- **`name`**：增强类型名称（用于文件名与训练侧筛选）
- **`prompt`**：该类型的自然语言编辑指令（会自动叠加 `prompt_prefix`）
- **`num_images`**（可选，默认 1）：同一类型每张图生成多少个变体（`v0/v1/...`）

输出文件命名规则：
- `data/nano_banana_pro/train/<image_id>__<name>__v<variant_idx>.<output_ext>`

---

### 输出目录与断点恢复（Manifest）

默认输出目录：`data/nano_banana_pro/train/`

脚本会生成：
- **增强图片**：`<image_id>__<aug_name>__v0.png` ...
- **manifest**：`manifest.csv`
- **日志**：`augment.log`

#### `manifest.csv` 列说明

- **`image_id`**：训练图片 ID（来自 `train.csv` 的 `sample_id` 去掉 `__target` 后缀）
- **`aug_name`**：增强类型名（来自 YAML 的 `augmentations[].name`）
- **`variant_idx`**：变体编号（0..num_images-1）
- **`src_image_path`**：源图片相对路径（来自 `train.csv` 的 `image_path`）
- **`image_path`**：生成图片路径（尽量写成相对 `data_root` 的路径，便于训练读取）
- **`status`**：`success` / `failed`
- **`error`**：失败原因（截断到 500 字符）
- **`ts`**：生成时间戳（秒）
- **`orig_w`/`orig_h`**：模型原始输出宽高（增强后续处理前）
- **`aspect_w`/`aspect_h`**：做完“强制长宽比”（pad/crop）后的宽高
- **`out_w`/`out_h`**：最终写盘图片宽高（若配置了 `postprocess.target_size` 则为该固定尺寸）
- **`pad_area_px`**：padding 新增的像素面积（`postprocess.mode=pad` 时通常 >0；crop 时为 0）
- **`area_scale`**：面积缩放系数 \( (aspect_w \cdot aspect_h)/(orig_w \cdot orig_h) \)。训练侧会用它对该样本的 `sample_area_m2` 做矫正。

#### 跳过/断点恢复规则

默认（不加 `--force`）：
- 若 `(image_id, aug_name, variant_idx)` 已在 manifest 里标记 `success` 且对应输出文件存在且非空：**跳过**
- 若输出文件已存在且非空（即使 manifest 里没有）：**跳过**

加 `--force`：会强制重新请求并覆写输出文件，同时继续追加 manifest 行。
（提示：旧版本生成的 manifest 会自动升级到新列；旧行的新增列会填默认值。若要补齐真实 `pad_area_px/area_scale`，需要 `--force` 重新生成对应图片。）

---

### 训练管线接入（开关：`configs/train.yaml`）

`configs/train.yaml` 新增：

```yaml
data:
  aigc_aug:
    enabled: false
    subdir: nano_banana_pro/train
    manifest: manifest.csv
    types: null
```

字段说明：
- **`enabled`**：是否启用 AIGC 增强数据接入训练（只影响 train，不影响 val）
- **`subdir`**：在 `data.root` 下的增强目录（默认 `data/nano_banana_pro/train`）
- **`manifest`**：manifest 文件名
- **`types`**：可选筛选，只接入这些 `aug_name`（`null`/空表示全部 success 行）

实现逻辑（简述）：
- DataModule 会在 `setup()` 完成 train/val split 后，读取 `manifest.csv`，将成功生成的增强图按 `image_id` join 回 `train_df`，并把这些行的 `image_path` 指向增强图，从而让 `PastureImageDataset` 直接读到增强图。
- 若 manifest 含 `area_scale`，则会为增强样本写入 `sample_area_m2 = base_area_m2 * area_scale`，从而在 `g -> g/m^2` 换算时做 per-image 矫正（类似 `train.yaml` 里 `width_m/length_m` 的面积逻辑，但这里是每张图不同）。

---

### 关于输出尺寸/长宽比（CSIRO: 2000×1000, W:H=2:1）

- **结论**：如果你使用的是“图生图编辑”（把原图作为输入），模型通常会**尽量跟随输入图的尺寸/长宽比**输出。
- **注意**：无论是 Gemini 的 `imageConfig.aspectRatio` 还是 OpenRouter 的 `image_config.aspect_ratio` 都只支持固定枚举（例如 `16:9/21:9/3:2/...`），**不包含 `2:1`**，因此不适合用它来“强制 2:1”。
- **本工具的保证方式**：在 `configs/nano_banana_pro_augment.yaml` 里提供了 `postprocess`，可对模型返回结果做 **pad（推荐，标签更安全）/crop**，并可选 **强制 resize 到 `[2000,1000]`**，从而保证严格 `2:1`。


