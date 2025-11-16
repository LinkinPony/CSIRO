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

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd

from torch import nn


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
    """
    Legacy helper: extract CLS+mean features for full images (non-MIR path).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

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
            chunk = features_cpu[i:i + max(1, batch_size)].to(device, non_blocking=True)
            out = head(chunk)
            preds_list.append(out.detach().cpu().float())
    return torch.cat(preds_list, dim=0) if preds_list else torch.empty((0, 3), dtype=torch.float32)


def _build_mir_tiling_cfg(cfg: Dict) -> Tuple[int, int, int, int, Optional[int]]:
    """
    Resolve MIR tiling config for the main dataset (typically 'csiro').
    Returns (tile_h, tile_w, stride_h, stride_w, max_tiles_per_image).
    """
    mir_cfg = dict(cfg.get("mir", {}))
    tiling = dict(mir_cfg.get("tiling", {}))
    ds_name = str(cfg.get("data", {}).get("dataset", "csiro"))
    ds_tiling = dict(tiling.get(ds_name, tiling.get("csiro", {})))
    size = ds_tiling.get("tile_size", None)
    if size is None or not isinstance(size, (list, tuple)) or len(size) != 2:
        # Fallback: single-tile covering the full resized image
        tile_w, tile_h = cfg["data"]["image_size"][0], cfg["data"]["image_size"][1]
    else:
        tile_w, tile_h = int(size[0]), int(size[1])
    stride = ds_tiling.get("tile_stride", size)
    if stride is None or not isinstance(stride, (list, tuple)) or len(stride) != 2:
        stride_w, stride_h = tile_w, tile_h
    else:
        stride_w, stride_h = int(stride[0]), int(stride[1])
    max_tiles = ds_tiling.get("max_tiles_per_image", None)
    max_tiles_int = int(max_tiles) if max_tiles is not None else None
    if max_tiles_int is not None and max_tiles_int <= 0:
        max_tiles_int = None
    # YAML is [W, H]; convert to (H, W)
    tile_h = max(1, int(tile_h))
    tile_w = max(1, int(tile_w))
    stride_h = max(1, int(stride_h))
    stride_w = max(1, int(stride_w))
    return tile_h, tile_w, stride_h, stride_w, max_tiles_int


def _extract_mir_tile_features_for_images(
    feature_extractor: nn.Module,
    dataset_root: str,
    image_paths: List[str],
    image_size: int,
    mean: List[float],
    std: List[float],
    batch_size: int,
    num_workers: int,
    tile_h: int,
    tile_w: int,
    stride_h: int,
    stride_w: int,
    max_tiles_per_image: Optional[int],
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    """
    Extract DINOv3 features for MIR tiles across all images.

    Returns:
        rels: list of image paths in order.
        tile_feats_cpu: (N_total, D) tile-level features on CPU.
        owners_cpu: (N_total,) long tensor mapping each tile to its image index.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = build_transforms(image_size=image_size, mean=mean, std=std)
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    feature_extractor.eval().to(device)

    rels: List[str] = []
    tile_feats_cpu: List[torch.Tensor] = []
    owners_cpu: List[torch.Tensor] = []

    global_index = 0
    with torch.inference_mode():
        for images, rel_paths in dl:
            images = images.to(device, non_blocking=True)  # (B,C,H,W)
            B, _, H, W = images.shape
            rels.extend(list(rel_paths))

            # Clamp tile size to actual image size (after resize)
            th = min(tile_h, H)
            tw = min(tile_w, W)
            sh = stride_h
            sw = stride_w

            tiles: List[torch.Tensor] = []
            owners: List[int] = []

            for b in range(B):
                img = images[b : b + 1]  # (1,C,H,W)
                positions: List[Tuple[int, int]] = []
                y = 0
                while y + th <= H:
                    x = 0
                    while x + tw <= W:
                        positions.append((y, x))
                        x += sw
                    y += sh
                if not positions:
                    # Fallback: use full image as a single tile
                    tiles.append(img)
                    owners.append(global_index + b)
                    continue
                if max_tiles_per_image is not None and len(positions) > max_tiles_per_image:
                    positions = positions[:max_tiles_per_image]
                for (yy, xx) in positions:
                    tiles.append(img[..., yy : yy + th, xx : xx + tw])
                    owners.append(global_index + b)

            if tiles:
                tile_batch = torch.cat(tiles, dim=0)  # (N_b,C,th,tw)
                feats = feature_extractor(tile_batch)
                tile_feats_cpu.append(feats.detach().cpu().float())
                owners_cpu.append(torch.tensor(owners, dtype=torch.long))

            global_index += B

    if not tile_feats_cpu:
        # Degenerate fallback: no tiles created; return empty tensors
        return rels, torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)

    tile_feats = torch.cat(tile_feats_cpu, dim=0)
    owners = torch.cat(owners_cpu, dim=0)
    return rels, tile_feats, owners


def _mir_pool_inference(
    tile_feats: torch.Tensor,
    tile_to_image: torch.Tensor,
    mir_meta: Dict,
    state: Dict,
) -> torch.Tensor:
    """
    Attention-based MIR pooling over tile features using weights stored in head state_dict.

    Args:
        tile_feats: (N_total, D)
        tile_to_image: (N_total,) long tensor mapping each tile to its image index.
        mir_meta: dict from head meta["mir"] (enabled, attn_hidden_dim, num_heads, token_normalize, instance_dropout).
        state: head state_dict containing MIR weights under keys:
            - "mir_ln.weight", "mir_ln.bias"
            - "mir_attn_proj.weight", "mir_attn_proj.bias"
            - "mir_attn_out.weight", "mir_attn_out.bias"
    Returns:
        bag_feats: (B, D)
    """
    if tile_feats.numel() == 0 or tile_feats.dim() != 2 or tile_to_image.numel() == 0:
        return tile_feats

    device = tile_feats.device
    N, D = tile_feats.shape
    B = int(tile_to_image.max().item()) + 1
    if B <= 0:
        return tile_feats.new_zeros((0, D))

    token_norm = str(mir_meta.get("token_normalize", "none")).lower()
    num_heads = int(mir_meta.get("num_heads", 1))
    num_heads = max(1, num_heads)

    # Load MIR weights if present; otherwise fall back to simple mean pooling.
    ln_w = state.get("mir_ln.weight", None)
    ln_b = state.get("mir_ln.bias", None)
    attn_proj_w = state.get("mir_attn_proj.weight", None)
    attn_proj_b = state.get("mir_attn_proj.bias", None)
    attn_out_w = state.get("mir_attn_out.weight", None)
    attn_out_b = state.get("mir_attn_out.bias", None)

    have_mir_weights = (
        attn_proj_w is not None
        and attn_out_w is not None
        and isinstance(attn_proj_w, torch.Tensor)
        and isinstance(attn_out_w, torch.Tensor)
    )
    if not have_mir_weights:
        # No MIR parameters available (e.g., legacy head); degrade to mean pooling.
        bag_feats = tile_feats.new_zeros((B, D))
        for b in range(B):
            mask = (tile_to_image == b)
            if not bool(mask.any()):
                continue
            h_b = tile_feats[mask]
            bag_feats[b] = h_b.mean(dim=0)
        return bag_feats

    h = tile_feats
    if token_norm == "layernorm" and isinstance(ln_w, torch.Tensor) and isinstance(ln_b, torch.Tensor):
        ln_w_d = ln_w.to(device=device, dtype=h.dtype)
        ln_b_d = ln_b.to(device=device, dtype=h.dtype)
        h = torch.layer_norm(h, normalized_shape=(D,), weight=ln_w_d, bias=ln_b_d)
    elif token_norm == "l2":
        h = torch.nn.functional.normalize(h, p=2.0, dim=-1)

    attn_proj_w_d = attn_proj_w.to(device=device, dtype=h.dtype)  # type: ignore[union-attr]
    attn_proj_b_d = attn_proj_b.to(device=device, dtype=h.dtype) if isinstance(attn_proj_b, torch.Tensor) else None
    attn_out_w_d = attn_out_w.to(device=device, dtype=h.dtype)  # type: ignore[union-attr]
    attn_out_b_d = attn_out_b.to(device=device, dtype=h.dtype) if isinstance(attn_out_b, torch.Tensor) else None

    # (N, H_dim)
    u = torch.tanh(torch.nn.functional.linear(h, attn_proj_w_d, attn_proj_b_d))
    # (N, num_heads)
    scores = torch.nn.functional.linear(u, attn_out_w_d, attn_out_b_d)

    bag_feats = tile_feats.new_zeros((B, D))
    for b in range(B):
        mask = (tile_to_image == b)
        if not bool(mask.any()):
            continue
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        h_b = h[idx]  # (T_b, D)
        scores_b = scores[idx]  # (T_b, num_heads)
        head_outputs: List[torch.Tensor] = []
        for k in range(num_heads):
            attn_raw = scores_b[:, k]
            attn = torch.nn.functional.softmax(attn_raw, dim=0)
            head_outputs.append(torch.sum(attn.unsqueeze(-1) * h_b, dim=0))
        bag_feats[b] = torch.stack(head_outputs, dim=0).mean(dim=0)

    return bag_feats


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

    # Data settings from config (single source of truth)
    image_size = _parse_image_size(cfg["data"]["image_size"])  # (H, W), e.g., (640, 640) or (640, 1280)
    mean = list(cfg["data"]["normalization"]["mean"])  # e.g., [0.485, 0.456, 0.406]
    std = list(cfg["data"]["normalization"]["std"])    # e.g., [0.229, 0.224, 0.225]
    target_bases = list(cfg["data"]["target_order"])    # e.g., [Dry_Clover_g, Dry_Dead_g, Dry_Green_g]
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

    # Discover head weights early to read optional PEFT payload
    head_weight_paths = discover_head_weight_paths(HEAD_WEIGHTS_PT_PATH)

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

    # 2) Use discovered heads (single or per-fold) and ensemble predictions by averaging

    preds_sum = torch.zeros((features_cpu.shape[0], 3), dtype=torch.float32)
    num_heads = 0
    for head_pt in head_weight_paths:
        # Read head state and meta (to detect log-scale setting if bundled)
        state, meta, _peft = load_head_state(head_pt)
        # Decide log-scale and effective Softplus for reg3
        log_scale_cfg = bool(cfg["model"].get("log_scale_targets", False))
        log_scale_meta = bool(meta.get("log_scale_targets", log_scale_cfg)) if isinstance(meta, dict) else log_scale_cfg
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
        use_softplus_eff = bool(cfg["model"]["head"].get("use_output_softplus", True)) and (not log_scale_meta) and (not zscore_enabled)
        head_module = build_head_layer(
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            num_outputs=3,
            head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
            head_activation=str(cfg["model"]["head"].get("activation", "relu")),
            dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
            use_output_softplus=use_softplus_eff,
        )
        # Filter out MIR-specific weights (mir_*) from the head state_dict when
        # constructing the plain MLP head for inference. MIR pooling is only
        # applied inside the training model and not reconstructed here.
        mlp_state = {k: v for k, v in state.items() if not k.startswith("mir_")}
        head_module.load_state_dict(mlp_state, strict=True)
        preds = predict_from_features(features_cpu=features_cpu, head=head_module, batch_size=batch_size)
        # Invert z-score, then log-scale if applicable, then convert to grams
        if zscore_enabled:
            preds = preds * reg3_std + reg3_mean  # type: ignore[operator]
        if log_scale_meta:
            preds = torch.expm1(preds).clamp_min(0.0)
        preds_sum += preds
        num_heads += 1

    if num_heads == 0:
        raise RuntimeError("No valid head weights found for inference.")
    avg_preds = (preds_sum / float(num_heads))
    # Convert g/m^2 to grams expected by submission format
    avg_preds = (avg_preds * float(area_m2)).tolist()

    # Build mapping from image path to base predictions
    image_to_base_preds: Dict[str, Tuple[float, float, float]] = {}
    for rel_path, vec in zip(rels_in_order, avg_preds):
        v0, v1, v2 = float(vec[0]), float(vec[1]), float(vec[2])
        image_to_base_preds[rel_path] = (v0, v1, v2)

    # Build submission
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])  # e.g., IDxxxx__Dry_Clover_g
        rel_path = str(r["image_path"])  # e.g., test/IDxxxx.jpg
        target_name = str(r["target_name"])  # one of 5
        v0, v1, v2 = image_to_base_preds.get(rel_path, (0.0, 0.0, 0.0))
        # Map base predictions by configured order
        base_map: Dict[str, float] = {}
        try:
            base_map[target_bases[0]] = v0
            base_map[target_bases[1]] = v1
            base_map[target_bases[2]] = v2
        except Exception:
            base_map = {}
        # Direct base prediction if target matches any base
        if target_name in base_map:
            value = base_map[target_name]
        elif target_name == "GDM_g":
            # GDM_g = Dry_Clover_g + Dry_Green_g
            value = base_map.get("Dry_Clover_g", 0.0) + base_map.get("Dry_Green_g", 0.0)
        elif target_name == "Dry_Total_g":
            # Prefer direct prediction if present; else sum of components when available
            if "Dry_Total_g" in base_map:
                value = base_map["Dry_Total_g"]
            else:
                value = base_map.get("Dry_Clover_g", 0.0) + base_map.get("Dry_Dead_g", 0.0) + base_map.get("Dry_Green_g", 0.0)
        elif target_name == "Dry_Dead_g":
            # Dry_Dead_g = Dry_Total_g - Dry_Clover_g - Dry_Green_g
            total = base_map.get("Dry_Total_g", None)
            clover = base_map.get("Dry_Clover_g", 0.0)
            green = base_map.get("Dry_Green_g", 0.0)
            if total is None:
                # Fallback: if no direct total, but individual base dead present (legacy case)
                value = base_map.get("Dry_Dead_g", 0.0)
            else:
                value = total - clover - green
        else:
            value = 0.0
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


if __name__ == "__main__":
    main()


