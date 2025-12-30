from __future__ import annotations

from dataclasses import dataclass


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


