from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class InferenceSettings:
    """
    Settings for the offline inference + submission pipeline.

    We keep this minimal and aligned with the user-editable variables at the top
    of `infer_and_submit_pt.py`.
    """

    head_weights_pt_path: str
    dino_weights_pt_path: str
    input_path: str
    output_submission_path: str

    # Repository root containing `configs/` and `src/`.
    project_dir: str

    # Optional: paths to vendored dependencies (added to sys.path by the entrypoint).
    dinov3_path: str = ""
    peft_path: str = ""

    # 2-GPU model-parallel inference for the very large ViT7B backbone.
    use_2gpu_model_parallel_for_vit7b: bool = True
    vit7b_mp_split_idx: int = 20
    vit7b_mp_dtype: str = "fp16"  # "fp16" | "fp32"

    # Inference batch size (decoupled from training config).
    infer_batch_size: int = 1

    # --------------------
    # Test-Time Augmentation (TTA)
    #
    # Notes:
    # - This repo's default eval transform is just Resize -> ToTensor -> Normalize.
    # - For the AugMix training policy, the only consistently used geometric aug is horizontal flip.
    # - We therefore default to a conservative TTA set: optional hflip and optional multi-scale resize.
    #
    # `tta_scales` are multipliers applied to cfg.data.image_size (H,W).
    # Example: (0.9, 1.0, 1.1). Sizes are rounded to multiples of the ViT patch size (16).
    # --------------------
    tta_enabled: bool = False
    tta_hflip: bool = True
    tta_vflip: bool = False
    tta_scales: tuple[float, ...] = (1.0,)

    # --------------------
    # MC Dropout (head-only) for uncertainty / robustness.
    #
    # IMPORTANT:
    # - We only enable Dropout layers inside the regression head during inference.
    # - Backbone stays in eval() and is only forwarded once per batch.
    # - We do NOT put the whole head in train() (to avoid BatchNorm / other stateful layers).
    # --------------------
    mc_dropout_enabled: bool = False
    mc_dropout_samples: int = 1
    # Optional base seed for reproducible MC sampling. When set (>=0), per-head seeds
    # are derived deterministically from this value.
    mc_dropout_seed: int = -1

    # --------------------
    # Post-training / Test-Time Training (TTT) on unlabeled images (e.g., test set)
    #
    # When enabled, the inference pipeline can:
    #  - load the specified head(+LoRA) weights
    #  - run a short unlabeled adaptation loop (A4: train LoRA + head)
    #  - save adapted head packages under `post_train_output_head_dir`
    #  - run inference using the adapted weights
    #
    # This is transductive; only enable when competition rules allow it.
    # --------------------
    post_train_enabled: bool = False
    # When true, NEVER run post-train inside the inference pipeline, even if configs/train.yaml
    # enables it. This is useful for evaluation scripts that need a clean "base vs post-train"
    # comparison without YAML accidentally triggering adaptation.
    post_train_force_disable: bool = False
    #
    # IMPORTANT (override semantics):
    # These fields are OPTIONAL overrides. When None / empty, inference should
    # use configs/train.yaml `post_train` config as the source of truth.
    #
    # This allows entrypoints (infer script, packaging, evaluation) to keep a single
    # POST_TRAIN_ENABLED flag while reusing train.yaml for all hyperparameters.
    # When false, reuse cached adapted weights if they already exist on disk.
    post_train_force: Optional[bool] = None
    # Directory to write adapted head packages into (mirrors `weights/head/` structure).
    # Example: "weights/head_post"
    post_train_output_head_dir: Optional[str] = None
    post_train_steps: Optional[int] = None
    post_train_batch_size: Optional[int] = None
    post_train_num_workers: Optional[int] = None
    post_train_lr_head: Optional[float] = None
    post_train_lr_lora: Optional[float] = None
    post_train_weight_reg3: Optional[float] = None
    post_train_weight_ratio: Optional[float] = None
    post_train_ema_enabled: Optional[bool] = None
    post_train_ema_decay: Optional[float] = None
    post_train_anchor_weight: Optional[float] = None
    post_train_log_every: Optional[int] = None
    post_train_seed: Optional[int] = None
    # Augmentation config override for post-train data loader (same schema as `data.augment` in train.yaml).
    post_train_augment: Optional[dict] = None


