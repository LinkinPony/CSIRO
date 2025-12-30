# ===== Required user variables =====
# Backward-compat: WEIGHTS_PT_PATH is ignored when HEAD_WEIGHTS_PT_PATH is provided.
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

# ===== TabPFN inference (optional) =====
# When enabled, this script will:
#   1) Load CSIRO `train.csv`, pivot to image-level rows
#   2) Extract **image-model head penultimate features** (pre-final-linear) for train + test images
#   3) Train TabPFN on the FULL train.csv (no CV)
#   4) Predict all test images and write `submission.csv`
#
# IMPORTANT:
# - TabPFN weights are loaded **LOCAL-ONLY** from TABPFN_WEIGHTS_CKPT_PATH.
# - No HuggingFace download / auth is attempted in this script.
TABPFN_ENABLED = True

# Optional: path to TabPFN python package source (offline-friendly).
# If you have tabpfn installed in the environment, you can ignore this.
TABPFN_PATH = "third_party/TabPFN/src"

# TabPFN weights path can be EITHER:
#  - a single checkpoint file (.ckpt), OR
#  - a directory containing one or more .ckpt files (the script will auto-select one).
#
# REQUIRED when TABPFN_ENABLED=True: local TabPFN 2.5 regressor checkpoint (.ckpt)
TABPFN_WEIGHTS_CKPT_PATH = "tabpfn_weights/tabpfn-v2.5-regressor-v2.5_default.ckpt"

# TabPFN runtime params
TABPFN_DEVICE = "auto"  # "auto"|"cpu"|"cuda"|"cuda:0"|...
TABPFN_N_ESTIMATORS = 8
TABPFN_FIT_MODE = "fit_preprocessors"
TABPFN_INFERENCE_PRECISION = "auto"
TABPFN_IGNORE_PRETRAINING_LIMITS = True
TABPFN_N_JOBS = 1  # MultiOutputRegressor parallelism (each output fits its own TabPFNRegressor)
TABPFN_ENABLE_TELEMETRY = False
TABPFN_MODEL_CACHE_DIR = ""  # optional; e.g. "/tmp/tabpfn_cache"

# Training data for TabPFN:
# - Default: automatically use <dataset_root>/train.csv, where dataset_root is resolved from INPUT_PATH.
# - Override: set to an explicit path to train.csv.
TABPFN_TRAIN_CSV_PATH = "data/train.csv"  # optional override

# Penultimate feature extraction settings
TABPFN_FEATURE_FUSION = "mean"  # "mean" | "concat" (concat can exceed TabPFN feature limits)
TABPFN_FEATURE_BATCH_SIZE = 8
TABPFN_FEATURE_NUM_WORKERS = 8
TABPFN_FEATURE_CACHE_PATH_TRAIN = ""  # optional .pt cache
TABPFN_FEATURE_CACHE_PATH_TEST = ""  # optional .pt cache

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
    _add_import_root(TABPFN_PATH, package_name="tabpfn")

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

    def _resolve_under_project_dir(p: str) -> str:
        """
        Resolve a possibly-relative path.

        Kaggle notebooks typically run from `/kaggle/working`, while the repo lives under
        `/kaggle/input/<dataset>/...`. Users often set PROJECT_DIR correctly but keep
        vendor paths like `third_party/dinov3` relative. This helper makes those paths
        behave as-if they were relative to PROJECT_DIR.
        """
        raw = str(p or "").strip()
        if not raw:
            return raw
        if os.path.isabs(raw) and os.path.exists(raw):
            return raw
        # First: resolve relative to current working dir (backward-compatible)
        cwd_abs = os.path.abspath(raw)
        if os.path.exists(cwd_abs):
            return cwd_abs
        # Fallback: resolve relative to PROJECT_DIR
        cand = os.path.join(project_dir_abs, raw)
        if os.path.exists(cand):
            return os.path.abspath(cand)
        return raw

    dinov3_path_resolved = _resolve_under_project_dir(str(DINOV3_PATH))
    peft_path_resolved = _resolve_under_project_dir(str(PEFT_PATH))
    tabpfn_path_resolved = _resolve_under_project_dir(str(TABPFN_PATH))

    # Re-add vendor roots using PROJECT_DIR-relative resolution (helps on Kaggle).
    _add_import_root(dinov3_path_resolved, package_name="dinov3")
    _add_import_root(peft_path_resolved, package_name="peft")
    _add_import_root(tabpfn_path_resolved, package_name="tabpfn")
    _add_import_root(project_dir_abs)

    from src.inference.settings import InferenceSettings  # noqa: E402

    settings = InferenceSettings(
        head_weights_pt_path=str(HEAD_WEIGHTS_PT_PATH),
        dino_weights_pt_path=str(DINO_WEIGHTS_PT_PATH),
        input_path=str(INPUT_PATH),
        output_submission_path=str(OUTPUT_SUBMISSION_PATH),
        project_dir=str(PROJECT_DIR),
        dinov3_path=str(dinov3_path_resolved),
        peft_path=str(peft_path_resolved),
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

    if bool(TABPFN_ENABLED):
        from src.inference.tabpfn import TabPFNSubmissionSettings, run_tabpfn_submission  # noqa: E402

        tabpfn_settings = TabPFNSubmissionSettings(
            weights_ckpt_path=str(TABPFN_WEIGHTS_CKPT_PATH),
            tabpfn_python_path=str(tabpfn_path_resolved),
            device=str(TABPFN_DEVICE),
            n_estimators=int(TABPFN_N_ESTIMATORS),
            fit_mode=str(TABPFN_FIT_MODE),
            inference_precision=str(TABPFN_INFERENCE_PRECISION),
            ignore_pretraining_limits=bool(TABPFN_IGNORE_PRETRAINING_LIMITS),
            n_jobs=int(TABPFN_N_JOBS),
            enable_telemetry=bool(TABPFN_ENABLE_TELEMETRY),
            model_cache_dir=str(TABPFN_MODEL_CACHE_DIR),
            train_csv_path=str(TABPFN_TRAIN_CSV_PATH),
            feature_fusion=str(TABPFN_FEATURE_FUSION),
            feature_batch_size=int(TABPFN_FEATURE_BATCH_SIZE),
            feature_num_workers=int(TABPFN_FEATURE_NUM_WORKERS),
            feature_cache_path_train=str(TABPFN_FEATURE_CACHE_PATH_TRAIN),
            feature_cache_path_test=str(TABPFN_FEATURE_CACHE_PATH_TEST),
        )
        run_tabpfn_submission(settings=settings, tabpfn=tabpfn_settings)
        return

    from src.inference.pipeline import run  # noqa: E402

    run(settings)


if __name__ == "__main__":
    main()


