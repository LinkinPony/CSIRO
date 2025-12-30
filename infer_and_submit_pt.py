# ===== Required user variables =====
# Backward-compat: WEIGHTS_PT_PATH is ignored when HEAD_WEIGHTS_PT_PATH is provided.
from pickle import TRUE


HEAD_WEIGHTS_PT_PATH = "weights/head/"  # regression head-only weights (.pt)
# Backbone weights path can be EITHER:
#  - a single weights file (.pt or .pth), OR
#  - a directory that contains the official DINOv3 weights files.
# In directory mode, the script will auto-select the correct file based on the backbone name.
DINO_WEIGHTS_PT_PATH = "dinov3_weights"  # file or directory containing DINOv3 weights
INPUT_PATH = "data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "submission.csv"
# IMPORTANT:
# Set this to the *repo root* that contains the `dinov3/` Python package folder.
# In this repository that is: "third_party/dinov3"
#
# Backward-compat: if you set it to ".../dinov3" (the package folder itself),
# the script will automatically add its parent to sys.path to avoid shadowing
# the Python stdlib `logging` module (dinov3 has a subpackage `dinov3.logging`).
DINOV3_PATH = "third_party/dinov3"
PEFT_PATH = "third_party/peft/src"  # path to peft source folder (contains peft/*)

# New: specify the project directory that contains both `configs/` and `src/` folders.
# Example: PROJECT_DIR = "/media/dl/dataset/Git/CSIRO"
PROJECT_DIR = "."

# ===== Multi-GPU model-parallel inference (Scheme B) =====
# When running the VERY large dinov3_vit7b16 backbone on 2x16GB GPUs (e.g., Kaggle T4),
# we split the transformer blocks across cuda:0 and cuda:1 and move the token activations
# across devices once at the split boundary.
USE_2GPU_MODEL_PARALLEL_FOR_VIT7B = True
VIT7B_MP_SPLIT_IDX = 20
VIT7B_MP_DTYPE = "fp16"  # one of: "fp16", "fp32"
# ===================================

# ===== Inference runtime settings (not read from YAML) =====
INFER_BATCH_SIZE = 1
# ===== MC Dropout (head-only) =====
# When enabled, we keep backbone in eval() and sample the regression head K times with Dropout active.
# This provides a simple predictive uncertainty estimate and sometimes improves robustness.
MC_DROPOUT_ENABLED = False
MC_DROPOUT_SAMPLES = 16  # typical: 8-32
# Optional base seed for reproducible MC sampling. Set to -1 to disable seeding.
MC_DROPOUT_SEED = 42
# ==========================================================

# ===== Test-Time Augmentation (TTA) =====
# Conservative defaults that match the repo's AugMix training policy:
# - Horizontal flip is the only consistently used geometric augmentation.
# - Optional multi-scale resize can be enabled, but increases compute cost.
TTA_ENABLED = False
TTA_HFLIP = True
TTA_VFLIP = True
# Scales are multipliers applied to cfg.data.image_size (H,W) and will be rounded to multiples of 16.
# Typical options: [1.0] (off), or [0.9, 1.0, 1.1] (heavier).
TTA_SCALES = [1.0]
# =======================================


import os
import sys


def _add_import_root(path: str, *, package_name: str | None = None) -> None:
    """
    Add a directory to sys.path such that `import <package_name>` works.

    - If `path` is a *repo root* containing `<package_name>/__init__.py`, we add `path`.
    - If `path` is the *package directory* itself (basename == package_name), we add its parent.

    This avoids accidental shadowing of stdlib modules (e.g., adding dinov3's package dir
    directly would expose `logging/` as a top-level package and break `import logging`).
    """
    p = os.path.abspath(path) if path else ""
    if not (p and os.path.isdir(p)):
        return

    root = p
    if package_name:
        pkg_init = os.path.join(p, package_name, "__init__.py")
        if os.path.isfile(pkg_init):
            root = p
        else:
            # If user passed the package dir itself, use its parent.
            if os.path.basename(p) == package_name and os.path.isfile(os.path.join(p, "__init__.py")):
                root = os.path.dirname(p)

    if root and os.path.isdir(root) and root not in sys.path:
        # Put vendor roots first so we prefer bundled dependencies over system installs.
        sys.path.insert(0, root)


def main() -> None:
    # Prefer local dinov3 / peft bundles when present (offline-friendly).
    _add_import_root(DINOV3_PATH, package_name="dinov3")
    _add_import_root(PEFT_PATH, package_name="peft")

    # Validate project directory and import project modules.
    project_dir_abs = os.path.abspath(PROJECT_DIR) if PROJECT_DIR else ""
    configs_dir = os.path.join(project_dir_abs, "configs")
    src_dir = os.path.join(project_dir_abs, "src")
    if not (project_dir_abs and os.path.isdir(project_dir_abs)):
        raise RuntimeError("PROJECT_DIR must point to the repository root containing `configs/` and `src/`.")
    if not os.path.isdir(configs_dir):
        raise RuntimeError(f"configs/ not found under PROJECT_DIR: {configs_dir}")
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"src/ not found under PROJECT_DIR: {src_dir}")
    _add_import_root(project_dir_abs)

    from src.inference.pipeline import run  # noqa: E402
    from src.inference.settings import InferenceSettings  # noqa: E402

    settings = InferenceSettings(
        head_weights_pt_path=str(HEAD_WEIGHTS_PT_PATH),
        dino_weights_pt_path=str(DINO_WEIGHTS_PT_PATH),
        input_path=str(INPUT_PATH),
        output_submission_path=str(OUTPUT_SUBMISSION_PATH),
        project_dir=str(PROJECT_DIR),
        dinov3_path=str(DINOV3_PATH),
        peft_path=str(PEFT_PATH),
        use_2gpu_model_parallel_for_vit7b=bool(USE_2GPU_MODEL_PARALLEL_FOR_VIT7B),
        vit7b_mp_split_idx=int(VIT7B_MP_SPLIT_IDX),
        vit7b_mp_dtype=str(VIT7B_MP_DTYPE),
        infer_batch_size=int(INFER_BATCH_SIZE),
        tta_enabled=bool(TTA_ENABLED),
        tta_hflip=bool(TTA_HFLIP),
        tta_vflip=bool(TTA_VFLIP),
        tta_scales=tuple(float(x) for x in (TTA_SCALES or [1.0])),
        mc_dropout_enabled=bool(MC_DROPOUT_ENABLED),
        mc_dropout_samples=int(MC_DROPOUT_SAMPLES),
        mc_dropout_seed=int(MC_DROPOUT_SEED),
    )
    run(settings)


if __name__ == "__main__":
    main()


