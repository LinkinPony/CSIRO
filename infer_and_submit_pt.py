# ===== Required user variables =====
# Backward-compat: WEIGHTS_PT_PATH is ignored when HEAD_WEIGHTS_PT_PATH is provided.
HEAD_WEIGHTS_PT_PATH = "weights/head/"  # regression head-only weights (.pt)
DINO_WEIGHTS_PT_PATH = "dinov3_weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pt"  # frozen DINOv3 weights (.pt)
INPUT_PATH = "data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "submission.csv"
DINOV3_PATH = "third_party/dinov3/dinov3"  # path to dinov3 source folder (contains dinov3/*)
PEFT_PATH = "third_party/peft/src"  # path to peft source folder (contains peft/*)

# New: specify the project directory that contains both `configs/` and `src/` folders.
# Example: PROJECT_DIR = "/media/dl/dataset/Git/CSIRO"
PROJECT_DIR = "."
# ===================================
# ==========================================================
# INFERENCE SCRIPT (UPDATED REQUIREMENTS)
# - Allowed to import this project's source code from `src/` and configuration from `configs/`.
# - Ensures single source of truth: model head settings, image transforms, etc. are read from YAML config.
# - A valid PROJECT_DIR must be provided, and it must contain both `configs/` and `src/`.
# - Inference requires two weights when using the new format:
#   1) DINOv3 backbone weights: DINO_WEIGHTS_PT_PATH (frozen, shared across runs)
#   2) Regression head weights: HEAD_WEIGHTS_PT_PATH (packaged as weights/head/infer_head.pt)
# - Legacy support: if HEAD_WEIGHTS_PT_PATH is empty, fall back to WEIGHTS_PT_PATH.
# ==========================================================


import os
import sys
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd

from torch import nn
import torch.nn.functional as F


import yaml

# ===== Add local dinov3 to import path (optional, for offline use) =====
_DINOV3_DIR = os.path.abspath(DINOV3_PATH) if DINOV3_PATH else ""
if _DINOV3_DIR and os.path.isdir(_DINOV3_DIR) and _DINOV3_DIR not in sys.path:
    sys.path.insert(0, _DINOV3_DIR)

# ===== Prefer vendored PEFT over system installation (if available) =====
_PEFT_DIR = os.path.abspath(PEFT_PATH) if PEFT_PATH else ""
if _PEFT_DIR and os.path.isdir(_PEFT_DIR) and _PEFT_DIR not in sys.path:
    sys.path.insert(0, _PEFT_DIR)

# ===== Validate project directory and import project modules =====
_PROJECT_DIR_ABS = os.path.abspath(PROJECT_DIR) if PROJECT_DIR else ""
_CONFIGS_DIR = os.path.join(_PROJECT_DIR_ABS, "configs")
_SRC_DIR = os.path.join(_PROJECT_DIR_ABS, "src")
if not (_PROJECT_DIR_ABS and os.path.isdir(_PROJECT_DIR_ABS)):
    raise RuntimeError("PROJECT_DIR must point to the repository root containing `configs/` and `src/`.")
if not os.path.isdir(_CONFIGS_DIR):
    raise RuntimeError(f"configs/ not found under PROJECT_DIR: {_CONFIGS_DIR}")
if not os.path.isdir(_SRC_DIR):
    raise RuntimeError(f"src/ not found under PROJECT_DIR: {_SRC_DIR}")
if _PROJECT_DIR_ABS not in sys.path:
    sys.path.insert(0, _PROJECT_DIR_ABS)

from src.models.head_builder import build_head_layer  # noqa: E402
from src.models.peft_integration import _import_peft  # noqa: E402
from src.data.augmentations import build_eval_transform  # noqa: E402


# ===== Weights loader (TorchScript supported, state_dict also supported) =====
# Allowlist PEFT types for PyTorch 2.6+ safe deserialization (weights_only=True)
try:
    from torch.serialization import add_safe_globals  # type: ignore
except Exception:
    add_safe_globals = None  # type: ignore

try:
    from peft.utils.peft_types import PeftType  # type: ignore
except Exception:
    try:
        # Ensure third_party/peft is importable if bundled
        _import_peft()
        from peft.utils.peft_types import PeftType  # type: ignore
    except Exception:
        PeftType = None  # type: ignore

if add_safe_globals is not None and PeftType is not None:  # type: ignore
    try:
        add_safe_globals([PeftType])  # type: ignore
    except Exception:
        pass

def load_model_or_state(pt_path: str) -> Tuple[Optional[nn.Module], Optional[dict], dict]:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Weights not found: {pt_path}")
    # 1) Try TorchScript (best for offline single-file inference)
    try:
        scripted = torch.jit.load(pt_path, map_location="cpu")
        return scripted, None, {"format": "torchscript"}
    except Exception:
        pass
    # 2) Fallback to torch.load objects (may be state_dict or checkpoint dict)
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return None, obj["state_dict"], obj.get("meta", {})
    if isinstance(obj, dict):
        # raw state_dict
        return None, obj, {}
    if isinstance(obj, nn.Module):
        # Pickled module (works only if class definitions are available)
        return obj, None, {"format": "pickled_module"}
    raise RuntimeError("Unsupported weights file format. Provide a TorchScript .pt for offline inference.")



def resolve_paths(input_path: str) -> Tuple[str, str]:
    if os.path.isdir(input_path):
        dataset_root = input_path
        test_csv = os.path.join(input_path, "test.csv")
    else:
        dataset_root = os.path.dirname(os.path.abspath(input_path))
        test_csv = input_path
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    return dataset_root, test_csv


class TestImageDataset(Dataset):
    def __init__(self, image_paths: List[str], root_dir: str, transform: T.Compose) -> None:
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        rel_path = self.image_paths[index]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, rel_path


def build_transforms(image_size: Tuple[int, int], mean: List[float], std: List[float]) -> T.Compose:
    return build_eval_transform(image_size=image_size, mean=mean, std=std)


def load_state_and_meta(pt_path: str):
    # Deprecated: kept for backward compatibility with older code paths
    model, state_dict, meta = load_model_or_state(pt_path)
    if model is not None:
        return model, meta
    return state_dict, meta


def load_head_state(pt_path: str) -> Tuple[dict, dict, Optional[dict]]:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Head weights not found: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {}), obj.get("peft", None)
    if isinstance(obj, dict):
        return obj, {}, obj.get("peft", None)
    raise RuntimeError("Unsupported head weights file format. Expect a dict with 'state_dict'.")


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)
        # Support PEFT-wrapped backbones (forward_features may live on base_model)
        try:
            forward_features = getattr(self.backbone, "forward_features", None)
            if forward_features is None and hasattr(self.backbone, "base_model"):
                forward_features = getattr(self.backbone.base_model, "forward_features", None)
            if forward_features is None:
                out = self.backbone(images)
                feats = out if isinstance(out, dict) else {"x_norm_clstoken": out}
            else:
                feats = forward_features(images)
        except Exception:
            feats = self.backbone.forward_features(images)
        # Build CLS + mean(patch) features
        cls = feats.get("x_norm_clstoken", None)
        if cls is None:
            raise RuntimeError("Backbone did not return 'x_norm_clstoken' in forward_features output")
        pt = None
        for k in ("x_norm_patchtokens", "x_norm_patch_tokens", "x_patch_tokens", "x_tokens"):
            if isinstance(feats, dict) and k in feats:
                pt = feats[k]
                break
        if pt is None and isinstance(feats, (list, tuple)) and len(feats) >= 2:
            pt = feats[1]
        if pt is None:
            raise RuntimeError("Backbone did not return patch tokens in forward_features output")
        if pt.dim() != 3:
            raise RuntimeError(f"Unexpected patch tokens shape: {tuple(pt.shape)}")
        patch_mean = pt.mean(dim=1)
        return torch.cat([cls, patch_mean], dim=-1)


class OfflineRegressor(nn.Module):
    def __init__(self, feature_extractor: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.head(features)


def load_config(project_dir: str) -> Dict:
    config_path = os.path.join(project_dir, "configs", "train.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config YAML not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
def _parse_image_size(value) -> Tuple[int, int]:
    try:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            w, h = int(value[0]), int(value[1])
            return (int(h), int(w))
        v = int(value)
        return (v, v)
    except Exception:
        v = int(value)
        return (v, v)



def discover_head_weight_paths(path: str) -> List[str]:
    # Accept single-file or directory containing per-fold heads
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        # Preferred: weights/head/fold_*/infer_head.pt
        fold_paths: List[str] = []
        try:
            for name in sorted(os.listdir(path)):
                if name.startswith("fold_"):
                    cand = os.path.join(path, name, "infer_head.pt")
                    if os.path.isfile(cand):
                        fold_paths.append(cand)
        except Exception:
            pass
        if fold_paths:
            return fold_paths
        # Fallback: weights/head/infer_head.pt under the directory
        single = os.path.join(path, "infer_head.pt")
        if os.path.isfile(single):
            return [single]
        # Fallback: any .pt directly under directory
        pts = [os.path.join(path, n) for n in os.listdir(path) if n.endswith('.pt')]
        pts.sort()
        if pts:
            return pts
    raise FileNotFoundError(f"Cannot find head weights at: {path}")


def _load_zscore_json_for_head(head_pt_path: str) -> Optional[dict]:
    """
    Try to locate z_score.json for a given head.
    Priority:
      1) Same directory as head .pt (e.g., weights/head/fold_i/z_score.json)
      2) Parent of head directory (e.g., weights/head/z_score.json)
      3) Parent of head parent (e.g., weights/z_score.json)
    """
    import json
    candidates: List[str] = []
    d = os.path.dirname(head_pt_path)
    candidates.append(os.path.join(d, "z_score.json"))
    parent = os.path.dirname(d)
    if parent and parent != d:
        candidates.append(os.path.join(parent, "z_score.json"))
        gp = os.path.dirname(parent)
        if gp and gp not in (parent, d):
            candidates.append(os.path.join(gp, "z_score.json"))
    for c in candidates:
        if os.path.isfile(c):
            try:
                with open(c, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                continue
    return None

def extract_features_for_images(
    feature_extractor: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
) -> Tuple[List[str], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    feature_extractor.eval().to(device)

    rels: List[str] = []
    feats_cpu: List[torch.Tensor] = []
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            feats = feature_extractor(images)
            feats_cpu.append(feats.detach().cpu().float())
            rels.extend(list(rel_paths))
    features = torch.cat(feats_cpu, dim=0) if feats_cpu else torch.empty((0, 0), dtype=torch.float32)
    return rels, features


def predict_from_features(features_cpu: torch.Tensor, head: nn.Module, batch_size: int) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.eval().to(device)
    N = features_cpu.shape[0]
    preds_list: List[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, N, max(1, batch_size)):
            chunk = features_cpu[i : i + max(1, batch_size)].to(device, non_blocking=True)
            out = head(chunk)
            preds_list.append(out.detach().cpu().float())
    # Do not assume a fixed output dimension; fall back to (0, 0) when empty.
    return torch.cat(preds_list, dim=0) if preds_list else torch.empty((0, 0), dtype=torch.float32)


def predict_from_features_mc_dropout(
    features_cpu: torch.Tensor,
    head: nn.Module,
    batch_size: int,
    num_samples: int,
) -> torch.Tensor:
    """
    Run Monte-Carlo dropout on the regression head only.

    Args:
        features_cpu: (N, D) features on CPU computed once by the frozen DINOv3 backbone.
        head: nn.Module containing dropout layers (kept in training mode to enable dropout).
        batch_size: mini-batch size for head-only forward passes.
        num_samples: number of stochastic forward passes to run.

    Returns:
        Tensor of shape (S, N, C) where:
          - S = num_samples
          - N = number of images
          - C = number of head outputs
    """
    if num_samples <= 0:
        preds = predict_from_features(features_cpu, head, batch_size)
        return preds.unsqueeze(0)  # (1, N, C)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = head.to(device)
    head.train()  # enable dropout during forward

    N = features_cpu.shape[0]
    samples: List[torch.Tensor] = []
    with torch.inference_mode():
        for _ in range(num_samples):
            preds_list: List[torch.Tensor] = []
            for i in range(0, N, max(1, batch_size)):
                chunk = features_cpu[i : i + max(1, batch_size)].to(device, non_blocking=True)
                out = head(chunk)
                preds_list.append(out.detach().cpu().float())
            preds = torch.cat(preds_list, dim=0) if preds_list else torch.empty((0, 0), dtype=torch.float32)
            samples.append(preds)

    if not samples:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    return torch.stack(samples, dim=0)  # (S, N, C)


def predict_for_images(
    model: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
) -> Dict[str, Tuple[float, float, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model.eval().to(device)

    preds: Dict[str, Tuple[float, float, float]] = {}
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            outputs = outputs.detach().cpu().float().tolist()
            for rel_path, vec in zip(rel_paths, outputs):
                v0, v1, v2 = float(vec[0]), float(vec[1]), float(vec[2])
                preds[rel_path] = (v0, v1, v2)
    return preds


def main():
    # Load configuration from project
    cfg = load_config(_PROJECT_DIR_ABS)

    # Reuse global seed from YAML for reproducible MC dropout and any other RNG usage.
    try:
        seed = int(cfg.get("seed", 42))
    except Exception:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        # Some backends/hardware combinations may not expose these flags; ignore in that case.
        pass

    # Data settings from config (single source of truth)
    image_size = _parse_image_size(cfg["data"]["image_size"])  # (H, W), e.g., (640, 640) or (640, 1280)
    mean = list(cfg["data"]["normalization"]["mean"])  # e.g., [0.485, 0.456, 0.406]
    std = list(cfg["data"]["normalization"]["std"])    # e.g., [0.229, 0.224, 0.225]
    target_bases = list(cfg["data"]["target_order"])    # legacy: base targets when using 3-d head
    # Dataset area (m^2) to convert g/m^2 (model output) back to grams for submission
    ds_name = str(cfg["data"].get("dataset", "csiro"))
    ds_map = dict(cfg["data"].get("datasets", {}))
    ds_info = dict(ds_map.get(ds_name, {}))
    try:
        width_m = float(ds_info.get("width_m", ds_info.get("width", 1.0)))
    except Exception:
        width_m = 1.0
    try:
        length_m = float(ds_info.get("length_m", ds_info.get("length", 1.0)))
    except Exception:
        length_m = 1.0
    try:
        area_m2 = float(ds_info.get("area_m2", width_m * length_m))
    except Exception:
        area_m2 = max(1.0, width_m * length_m if (width_m > 0.0 and length_m > 0.0) else 1.0)
    if not (area_m2 > 0.0):
        area_m2 = 1.0
    batch_size = int(cfg["data"].get("batch_size", 32))
    num_workers = int(cfg["data"].get("num_workers", 4))

    # MC dropout configuration (head-only, backbone always frozen), controlled purely via YAML.
    mc_cfg = dict(cfg.get("mc_dropout", {}) or {})
    mc_enabled = bool(mc_cfg.get("enabled", False))
    mc_num_samples = int(mc_cfg.get("num_samples", 0) or 0)
    debug_mode = bool(mc_cfg.get("debug", False))
    if not mc_enabled or mc_num_samples <= 1:
        mc_enabled = False
        mc_num_samples = 0

    # Read test.csv
    dataset_root, test_csv = resolve_paths(INPUT_PATH)
    df = pd.read_csv(test_csv)
    if not {"sample_id", "image_path", "target_name"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: sample_id, image_path, target_name")
    unique_image_paths = df["image_path"].astype(str).unique().tolist()

    # Build model and load weights (supports single head or k-fold heads under a directory)
    # Strictly offline path: require both backbone and head weights
    if not HEAD_WEIGHTS_PT_PATH:
        raise FileNotFoundError("HEAD_WEIGHTS_PT_PATH must be set to a valid head file or directory.")
    if not (DINO_WEIGHTS_PT_PATH and os.path.isfile(DINO_WEIGHTS_PT_PATH)):
        raise FileNotFoundError("DINO_WEIGHTS_PT_PATH must be set to a valid backbone .pt file.")

    # Import correct DINOv3 constructor based on config
    try:
        backbone_name = str(cfg["model"]["backbone"]).strip()
        if backbone_name == "dinov3_vith16plus":
            from dinov3.hub.backbones import dinov3_vith16plus as _make_backbone  # type: ignore
        elif backbone_name == "dinov3_vitl16":
            from dinov3.hub.backbones import dinov3_vitl16 as _make_backbone  # type: ignore
        else:
            raise ImportError(f"Unsupported backbone in config: {backbone_name}")
    except Exception:
        raise ImportError(
            "dinov3 is not available locally. Ensure DINOV3_PATH points to third_party/dinov3/dinov3."
        )

    # Discover head weights early and read meta to determine head format
    head_weight_paths = discover_head_weight_paths(HEAD_WEIGHTS_PT_PATH)

    # Inspect first head file to infer packed head shape (main + optional ratio outputs)
    first_state, first_meta, _ = load_head_state(head_weight_paths[0])
    if not isinstance(first_meta, dict):
        first_meta = {}
    num_outputs_main = int(first_meta.get("num_outputs_main", first_meta.get("num_outputs", 3)))
    num_outputs_ratio = int(first_meta.get("num_outputs_ratio", 0))
    head_total_outputs = int(first_meta.get("head_total_outputs", num_outputs_main + num_outputs_ratio))
    ratio_components = first_meta.get("ratio_components", [])
    is_ratio_format = num_outputs_ratio > 0 and head_total_outputs == (num_outputs_main + num_outputs_ratio)

    # Build backbone architecture locally (no torch.hub), then load state_dict
    backbone = _make_backbone(pretrained=False)
    dino_state = torch.load(DINO_WEIGHTS_PT_PATH, map_location="cpu")
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]
    try:
        backbone.load_state_dict(dino_state, strict=True)
    except Exception:
        backbone.load_state_dict(dino_state, strict=False)

    # If any head carries a PEFT payload, inject LoRA adapters into the backbone and load adapter weights
    try:
        # Load first head file to inspect for peft payload
        peft_payload: Optional[dict] = None
        if head_weight_paths:
            _obj = torch.load(head_weight_paths[0], map_location="cpu")
            if isinstance(_obj, dict) and ("peft" in _obj):
                peft_payload = _obj.get("peft")
        if peft_payload is not None and isinstance(peft_payload, dict):
            peft_cfg_dict = peft_payload.get("config", None)
            peft_state = peft_payload.get("state_dict", None)
            if peft_cfg_dict and peft_state:
                # Import PEFT lazily and wrap backbone
                try:
                    from peft.config import PeftConfig  # type: ignore
                    from peft.mapping_func import get_peft_model  # type: ignore
                    from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                except Exception:
                    # Fallback via our integration helper to ensure third_party path is added
                    LoraConfig, get_peft_model_alt, _, _ = _import_peft()  # noqa: F841
                    from peft.config import PeftConfig  # type: ignore
                    from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
                    get_peft_model = get_peft_model_alt  # type: ignore

                peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
                backbone = get_peft_model(backbone, peft_config)
                # Load adapter weights into wrapped model
                set_peft_model_state_dict(backbone, peft_state, adapter_name="default")
                backbone.eval()
    except Exception as _e:
        # If PEFT injection fails, continue without adapters
        print(f"[WARN] PEFT injection skipped: {_e}")

    feature_extractor = DinoV3FeatureExtractor(backbone)

    # 1) Extract DINOv3 features ONCE for all images
    rels_in_order, features_cpu = extract_features_for_images(
        feature_extractor=feature_extractor,
        dataset_root=dataset_root,
        image_paths=unique_image_paths,
        image_size=image_size,
        mean=mean,
        std=std,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # 2) Use discovered heads (single or per-fold) and ensemble predictions.
    #    When mc_dropout.enabled is set in the YAML, we keep the backbone frozen and
    #    run multiple stochastic passes through the head with dropout enabled.

    N = features_cpu.shape[0]
    image_to_components: Dict[str, Dict[str, float]] = {}

    if mc_enabled:
        # MC-dropout statistics over all stochastic passes across all heads.
        if is_ratio_format:
            # 5 physical outputs: [Dry_Total_g, Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g]
            mc_num_outputs = 5
        else:
            # Legacy head: base targets in grams
            mc_num_outputs = num_outputs_main
        mc_sum = torch.zeros((N, mc_num_outputs), dtype=torch.float32)
        mc_sumsq = torch.zeros_like(mc_sum)
        mc_total = 0
        mc_samples_list: List[torch.Tensor] = []
    else:
        if is_ratio_format:
            # New format: head outputs [main_reg, ratio_logits...]
            preds_main_sum = torch.zeros((N, num_outputs_main), dtype=torch.float32)
            preds_ratio_sum = torch.zeros((N, num_outputs_ratio), dtype=torch.float32)
        else:
            # Legacy format: 3-d head for base targets
            preds_sum = torch.zeros((N, num_outputs_main), dtype=torch.float32)
        num_heads = 0

    for head_pt in head_weight_paths:
        # Read head state and meta (to detect log-scale setting if bundled)
        state, meta, _peft = load_head_state(head_pt)
        if not isinstance(meta, dict):
            meta = {}
        # Decide log-scale for main reg3 outputs
        log_scale_cfg = bool(cfg["model"].get("log_scale_targets", False))
        log_scale_meta = bool(meta.get("log_scale_targets", log_scale_cfg))
        # Load z-score for this head (optional)
        zscore = _load_zscore_json_for_head(head_pt)
        reg3_mean = None
        reg3_std = None
        if isinstance(zscore, dict) and "reg3" in zscore:
            try:
                reg3_mean = torch.tensor(zscore["reg3"]["mean"], dtype=torch.float32)
                reg3_std = torch.tensor(zscore["reg3"]["std"], dtype=torch.float32).clamp_min(1e-8)
            except Exception:
                reg3_mean, reg3_std = None, None
        zscore_enabled = reg3_mean is not None and reg3_std is not None

        # For packed heads we always export without a terminal Softplus.
        # Prefer the activation used during training (stored in meta), and fall back
        # to config only if missing for backward compatibility.
        head_activation = str(meta.get("head_activation", cfg["model"]["head"].get("activation", "relu")))
        head_module = build_head_layer(
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            num_outputs=head_total_outputs if is_ratio_format else num_outputs_main,
            head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
            head_activation=head_activation,
            dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
            use_output_softplus=False,
        )
        head_module.load_state_dict(state, strict=True)

        if mc_enabled:
            # MC dropout: (S, N, C) samples in physical or latent space depending on head type.
            preds_samples_all = predict_from_features_mc_dropout(
                features_cpu=features_cpu,
                head=head_module,
                batch_size=batch_size,
                num_samples=mc_num_samples,
            )
            if preds_samples_all.ndim != 3 or preds_samples_all.shape[1] != N:
                continue

            if is_ratio_format:
                # Split main regression and ratio logits for each MC sample
                preds_main_s = preds_samples_all[:, :, :num_outputs_main]  # (S, N, num_outputs_main)
                preds_ratio_s = preds_samples_all[:, :, num_outputs_main : num_outputs_main + num_outputs_ratio]  # (S, N, num_outputs_ratio)

                # Invert z-score / log-scale in g/m^2 space for each sample
                if zscore_enabled and reg3_mean is not None and reg3_std is not None:
                    mean_view = reg3_mean[:num_outputs_main].view(1, 1, -1)
                    std_view = reg3_std[:num_outputs_main].view(1, 1, -1)
                    preds_main_s = preds_main_s * std_view + mean_view
                if log_scale_meta:
                    preds_main_s = torch.expm1(preds_main_s).clamp_min(0.0)

                # Convert to grams and decompose into 5D physical components per sample
                main_g_s = preds_main_s * float(area_m2)  # (S, N, 1)
                ratio_logits_s = preds_ratio_s  # (S, N, 3)
                p_ratio_s = F.softmax(ratio_logits_s, dim=-1)  # (S, N, 3)

                comp_g_s = p_ratio_s * main_g_s  # (S, N, 3)
                clover_g_s = comp_g_s[:, :, 0]
                dead_g_s = comp_g_s[:, :, 1]
                green_g_s = comp_g_s[:, :, 2]
                gdm_g_s = clover_g_s + green_g_s
                total_g_s = main_g_s.squeeze(-1)  # (S, N)

                # Stack as [Dry_Total_g, Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g]
                samples_phys = torch.stack(
                    [total_g_s, clover_g_s, dead_g_s, green_g_s, gdm_g_s],
                    dim=-1,
                )  # (S, N, 5)
            else:
                # Legacy: only main regression outputs, then convert to grams.
                preds_main_s = preds_samples_all  # (S, N, num_outputs_main)
                if zscore_enabled and reg3_mean is not None and reg3_std is not None:
                    mean_view = reg3_mean[:num_outputs_main].view(1, 1, -1)
                    std_view = reg3_std[:num_outputs_main].view(1, 1, -1)
                    preds_main_s = preds_main_s * std_view + mean_view
                if log_scale_meta:
                    preds_main_s = torch.expm1(preds_main_s).clamp_min(0.0)
                samples_phys = preds_main_s * float(area_m2)  # (S, N, num_outputs_main)

            # Aggregate statistics over MC samples for this head
            mc_sum += samples_phys.sum(dim=0)
            mc_sumsq += (samples_phys * samples_phys).sum(dim=0)
            mc_total += samples_phys.shape[0]

            if debug_mode:
                mc_samples_list.append(samples_phys)
        else:
            # Deterministic single-pass head inference (original behaviour)
            preds_all = predict_from_features(features_cpu=features_cpu, head=head_module, batch_size=batch_size)

            if is_ratio_format:
                preds_main = preds_all[:, :num_outputs_main]
                preds_ratio = preds_all[:, num_outputs_main : num_outputs_main + num_outputs_ratio]
            else:
                preds_main = preds_all
                preds_ratio = None

            # Invert z-score and log-scale for main reg3 outputs only, then convert to grams
            if zscore_enabled:
                preds_main = preds_main * reg3_std[:num_outputs_main] + reg3_mean[:num_outputs_main]  # type: ignore[index]
            if log_scale_meta:
                preds_main = torch.expm1(preds_main).clamp_min(0.0)

            if is_ratio_format:
                preds_main_sum += preds_main
                preds_ratio_sum += preds_ratio  # type: ignore[operator]
            else:
                preds_sum += preds_main
            num_heads += 1

    if mc_enabled:
        if mc_total == 0:
            raise RuntimeError("MC dropout inference failed: no valid samples were produced.")

        mean_phys = mc_sum / float(mc_total)
        var_phys = mc_sumsq / float(mc_total) - mean_phys * mean_phys
        var_phys = torch.clamp(var_phys, min=0.0)
        std_phys = torch.sqrt(var_phys + 1e-12)

        if is_ratio_format:
            # mean_phys: (N, 5) ordered as [Dry_Total_g, Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g]
            for rel_path, vec in zip(rels_in_order, mean_phys.tolist()):
                t, c, d, g, gdm_val = vec
                image_to_components[rel_path] = {
                    "Dry_Total_g": float(t),
                    "Dry_Clover_g": float(c),
                    "Dry_Dead_g": float(d),
                    "Dry_Green_g": float(g),
                    "GDM_g": float(gdm_val),
                }

            if debug_mode and mc_samples_list:
                all_samples = torch.cat(mc_samples_list, dim=0)  # (M, N, 5)
                output_names = ["Dry_Total_g", "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g"]
                debug_path = os.path.splitext(os.path.abspath(OUTPUT_SUBMISSION_PATH))[0] + "_mc_dropout_debug.json"
                try:
                    _write_mc_dropout_debug_csv(
                        csv_path=debug_path,
                        rel_paths=rels_in_order,
                        all_samples=all_samples,
                        mean=mean_phys,
                        var=var_phys,
                        std=std_phys,
                        output_names=output_names,
                    )
                    print(f"MC dropout debug CSV written to: {debug_path}")
                except Exception as e:
                    print(f"[WARN] Failed to write MC dropout debug CSV: {e}")
        else:
            # Legacy head: mean predictions in grams for base targets, then derive 5D components.
            mean_main_g = mean_phys  # (N, num_outputs_main)
            image_to_base_preds: Dict[str, Tuple[float, ...]] = {}
            for rel_path, vec in zip(rels_in_order, mean_main_g.tolist()):
                image_to_base_preds[rel_path] = tuple(float(v) for v in vec)

            for rel_path, vec in image_to_base_preds.items():
                base_map: Dict[str, float] = {}
                try:
                    for idx, name in enumerate(target_bases):
                        if idx < len(vec):
                            base_map[name] = vec[idx]
                except Exception:
                    base_map = {}
                total = base_map.get("Dry_Total_g", None)
                clover = base_map.get("Dry_Clover_g", None)
                dead = base_map.get("Dry_Dead_g", None)
                green = base_map.get("Dry_Green_g", None)
                if total is None:
                    total = (clover or 0.0) + (dead or 0.0) + (green or 0.0)
                if dead is None and total is not None and clover is not None and green is not None:
                    dead = total - clover - green
                if clover is None:
                    clover = 0.0
                if dead is None:
                    dead = 0.0
                if green is None:
                    green = 0.0
                gdm_val = clover + green
                image_to_components[rel_path] = {
                    "Dry_Total_g": float(total),
                    "Dry_Clover_g": float(clover),
                    "Dry_Dead_g": float(dead),
                    "Dry_Green_g": float(green),
                    "GDM_g": float(gdm_val),
                }

            if debug_mode and mc_samples_list:
                all_samples = torch.cat(mc_samples_list, dim=0)  # (M, N, num_outputs_main)
                output_names = [str(n) for n in target_bases[:num_outputs_main]]
                debug_path = os.path.splitext(os.path.abspath(OUTPUT_SUBMISSION_PATH))[0] + "_mc_dropout_debug.json"
                try:
                    _write_mc_dropout_debug_csv(
                        csv_path=debug_path,
                        rel_paths=rels_in_order,
                        all_samples=all_samples,
                        mean=mean_main_g,
                        var=var_phys,
                        std=std_phys,
                        output_names=output_names,
                    )
                    print(f"MC dropout debug CSV written to: {debug_path}")
                except Exception as e:
                    print(f"[WARN] Failed to write MC dropout debug CSV: {e}")
    else:
        if num_heads == 0:
            raise RuntimeError("No valid head weights found for inference.")

        if is_ratio_format:
            avg_main_gm2 = preds_main_sum / float(num_heads)  # (N,1)
            avg_ratio_logits = preds_ratio_sum / float(num_heads)  # (N,3)
            # Convert main reg output (g/m^2) to grams for Dry_Total_g
            avg_main_g = avg_main_gm2 * float(area_m2)
            # Ratio distribution over (Dry_Clover_g, Dry_Dead_g, Dry_Green_g)
            p_ratio = F.softmax(avg_ratio_logits, dim=-1)
            total_g = avg_main_g.squeeze(-1)
            clover_g = total_g * p_ratio[:, 0]
            dead_g = total_g * p_ratio[:, 1]
            green_g = total_g * p_ratio[:, 2]
            gdm_g = clover_g + green_g

            # Build mapping from image path to full 5D components
            for rel_path, t, c, d, g, gdm_val in zip(
                rels_in_order,
                total_g.tolist(),
                clover_g.tolist(),
                dead_g.tolist(),
                green_g.tolist(),
                gdm_g.tolist(),
            ):
                image_to_components[rel_path] = {
                    "Dry_Total_g": float(t),
                    "Dry_Clover_g": float(c),
                    "Dry_Dead_g": float(d),
                    "Dry_Green_g": float(g),
                    "GDM_g": float(gdm_val),
                }
        else:
            # Legacy 3-d head: outputs base targets in config-defined order, then derive 5D components.
            avg_preds = (preds_sum / float(num_heads))
            avg_preds = (avg_preds * float(area_m2)).tolist()
            image_to_base_preds: Dict[str, Tuple[float, ...]] = {}
            for rel_path, vec in zip(rels_in_order, avg_preds):
                image_to_base_preds[rel_path] = tuple(float(v) for v in vec)

            for rel_path, vec in image_to_base_preds.items():
                base_map: Dict[str, float] = {}
                try:
                    for idx, name in enumerate(target_bases):
                        base_map[name] = vec[idx]
                except Exception:
                    base_map = {}
                # Derive full 5D components from base_map using legacy rules
                total = base_map.get("Dry_Total_g", None)
                clover = base_map.get("Dry_Clover_g", None)
                dead = base_map.get("Dry_Dead_g", None)
                green = base_map.get("Dry_Green_g", None)
                if total is None:
                    # Fallback: infer Dry_Total_g if not directly predicted
                    total = (clover or 0.0) + (dead or 0.0) + (green or 0.0)
                if dead is None and total is not None and clover is not None and green is not None:
                    dead = total - clover - green
                if clover is None:
                    clover = 0.0
                if dead is None:
                    dead = 0.0
                if green is None:
                    green = 0.0
                gdm_val = clover + green
                image_to_components[rel_path] = {
                    "Dry_Total_g": float(total),
                    "Dry_Clover_g": float(clover),
                    "Dry_Dead_g": float(dead),
                    "Dry_Green_g": float(green),
                    "GDM_g": float(gdm_val),
                }

    # Build submission
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])  # e.g., IDxxxx__Dry_Clover_g
        rel_path = str(r["image_path"])  # e.g., test/IDxxxx.jpg
        target_name = str(r["target_name"])  # one of 5
        comps = image_to_components.get(rel_path, {})
        value = comps.get(target_name, 0.0)
        # Clamp final physical predictions to be non-negative for submission
        value = max(0.0, float(value))
        rows.append((sample_id, value))

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_SUBMISSION_PATH))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(OUTPUT_SUBMISSION_PATH, "w", encoding="utf-8") as f:
        f.write("sample_id,target\n")
        for sample_id, value in rows:
            f.write(f"{sample_id},{value}\n")
    print(f"Submission written to: {OUTPUT_SUBMISSION_PATH}")


def _write_mc_dropout_debug_csv(
    csv_path: str,
    rel_paths: List[str],
    all_samples: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    std: torch.Tensor,
    output_names: List[str],
) -> None:
    """
    Save MC dropout per-sample draws and summary statistics into a single JSON file.

    Each row corresponds to one image (identified by image_path). For each output
    dimension we record:
      - mean_{name}, var_{name}, std_{name}
      - mc{k}_{name} for k in [0, M-1] (per-draw predictions)
    """
    import numpy as np  # noqa: F401  # kept for potential future extensions

    M, N, C = all_samples.shape
    if N != len(rel_paths) or C != len(output_names):
        raise ValueError("Mismatch between samples, paths, and output names for MC dropout debug export.")

    mean_np = mean.detach().cpu().numpy()
    var_np = var.detach().cpu().numpy()
    std_np = std.detach().cpu().numpy()
    samples_np = all_samples.detach().cpu().numpy()

    rows: List[Dict[str, object]] = []
    for idx, rel_path in enumerate(rel_paths):
        row: Dict[str, object] = {"image_path": rel_path}
        for c_idx, name in enumerate(output_names):
            row[f"mean_{name}"] = float(mean_np[idx, c_idx])
            row[f"var_{name}"] = float(var_np[idx, c_idx])
            row[f"std_{name}"] = float(std_np[idx, c_idx])
        for m in range(M):
            for c_idx, name in enumerate(output_names):
                row[f"mc{m}_{name}"] = float(samples_np[m, idx, c_idx])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    # Write as JSON array of records for easier downstream parsing.
    df.to_json(csv_path, orient="records", indent=2)


if __name__ == "__main__":
    main()


