# ===== Required user variables =====
# Backward-compat: WEIGHTS_PT_PATH is ignored when HEAD_WEIGHTS_PT_PATH is provided.
HEAD_WEIGHTS_PT_PATH = "weights/head/"  # regression head-only weights (.pt)
DINO_WEIGHTS_PT_PATH = "dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pt"  # frozen DINOv3 weights (.pt)
INPUT_PATH = "data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "submission.csv"
DINOV3_PATH = "third_party/dinov3/dinov3"  # path to dinov3 source folder (contains dinov3/*)

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

try:
    from dinov3.hub.backbones import dinov3_vitl16 as _dinov3_vitl16  # noqa: F401
except Exception:
    _dinov3_vitl16 = None  # noqa: F401

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


# ===== Weights loader (TorchScript supported, state_dict also supported) =====
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
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])


def load_state_and_meta(pt_path: str):
    # Deprecated: kept for backward compatibility with older code paths
    model, state_dict, meta = load_model_or_state(pt_path)
    if model is not None:
        return model, meta
    return state_dict, meta


def load_head_state(pt_path: str) -> Tuple[dict, dict]:
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Head weights not found: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {})
    if isinstance(obj, dict):
        return obj, {}
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
        feats = self.backbone.forward_features(images)
        return feats["x_norm_clstoken"]


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
            chunk = features_cpu[i:i + max(1, batch_size)].to(device, non_blocking=True)
            out = head(chunk)
            preds_list.append(out.detach().cpu().float())
    return torch.cat(preds_list, dim=0) if preds_list else torch.empty((0, 3), dtype=torch.float32)


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
                dc, dd, dg = float(vec[0]), float(vec[1]), float(vec[2])
                preds[rel_path] = (dc, dd, dg)
    return preds


def main():
    # Load configuration from project
    cfg = load_config(_PROJECT_DIR_ABS)

    # Data settings from config (single source of truth)
    image_size = _parse_image_size(cfg["data"]["image_size"])  # (H, W), e.g., (640, 640) or (640, 1280)
    mean = list(cfg["data"]["normalization"]["mean"])  # e.g., [0.485, 0.456, 0.406]
    std = list(cfg["data"]["normalization"]["std"])    # e.g., [0.229, 0.224, 0.225]
    target_bases = list(cfg["data"]["target_order"])    # e.g., [Dry_Clover_g, Dry_Dead_g, Dry_Green_g]
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

    if _dinov3_vitl16 is None:
        raise ImportError(
            "dinov3 is not available locally. Ensure DINOV3_PATH points to third_party/dinov3/dinov3."
        )

    # Build backbone architecture locally (no torch.hub), then load state_dict
    backbone = _dinov3_vitl16(pretrained=False)
    dino_state = torch.load(DINO_WEIGHTS_PT_PATH, map_location="cpu")
    if isinstance(dino_state, dict) and "state_dict" in dino_state:
        dino_state = dino_state["state_dict"]
    try:
        backbone.load_state_dict(dino_state, strict=True)
    except Exception:
        backbone.load_state_dict(dino_state, strict=False)

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

    # 2) Discover heads (single or per-fold) and ensemble predictions by averaging
    head_weight_paths = discover_head_weight_paths(HEAD_WEIGHTS_PT_PATH)

    preds_sum = torch.zeros((features_cpu.shape[0], 3), dtype=torch.float32)
    num_heads = 0
    for head_pt in head_weight_paths:
        head_module = build_head_layer(
            embedding_dim=int(cfg["model"]["embedding_dim"]),
            num_outputs=3,
            head_hidden_dims=list(cfg["model"]["head"].get("hidden_dims", [512, 256])),
            head_activation=str(cfg["model"]["head"].get("activation", "relu")),
            dropout=float(cfg["model"]["head"].get("dropout", 0.0)),
            use_output_softplus=bool(cfg["model"]["head"].get("use_output_softplus", True)),
        )
        state, _ = load_head_state(head_pt)
        head_module.load_state_dict(state, strict=True)
        preds = predict_from_features(features_cpu=features_cpu, head=head_module, batch_size=batch_size)
        preds_sum += preds
        num_heads += 1

    if num_heads == 0:
        raise RuntimeError("No valid head weights found for inference.")
    avg_preds = (preds_sum / float(num_heads)).tolist()

    # Build mapping from image path to base predictions
    image_to_base_preds: Dict[str, Tuple[float, float, float]] = {}
    for rel_path, vec in zip(rels_in_order, avg_preds):
        dc, dd, dg = float(vec[0]), float(vec[1]), float(vec[2])
        image_to_base_preds[rel_path] = (dc, dd, dg)

    # Build submission
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])  # e.g., IDxxxx__Dry_Clover_g
        rel_path = str(r["image_path"])  # e.g., test/IDxxxx.jpg
        target_name = str(r["target_name"])  # one of 5
        dc, dd, dg = image_to_base_preds.get(rel_path, (0.0, 0.0, 0.0))
        if target_name == target_bases[0]:  # Dry_Clover_g
            value = dc
        elif target_name == target_bases[1]:  # Dry_Dead_g
            value = dd
        elif target_name == target_bases[2]:  # Dry_Green_g
            value = dg
        elif target_name == "GDM_g":
            value = dc + dg
        elif target_name == "Dry_Total_g":
            value = dc + dd + dg
        else:
            value = 0.0
        rows.append((sample_id, float(value)))

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


