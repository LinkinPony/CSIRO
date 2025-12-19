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


