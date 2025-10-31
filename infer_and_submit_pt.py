# ===== Required user variables =====
WEIGHTS_PT_PATH = "weights/best.pt"  # exported by export_ckpt_to_pt.py
INPUT_PATH = "data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "submission.csv"
DINOV3_PATH = "third_party/dinov3/dinov3"  # path to dinov3 source folder (contains dinov3/*)
# ===================================
# ==========================================================
# STRICT OFFLINE INFERENCE SCRIPT (READ ME CAREFULLY)
# - Only one weight file is allowed: WEIGHTS_PT_PATH, exported by export_ckpt_to_pt.py
# - Strictly offline: NO hub/network downloads allowed anywhere
# - Only code allowed to be imported outside this file is the dinov3 library
#   from DINOV3_PATH. Do NOT import any other local project code.
# - This script reconstructs the MLP head used during training from the
#   configuration below. If you changed training head settings, update here too.
# ==========================================================



# ===== Model/data configuration (copy of required train.yaml parts) =====
# Backbone and embedding
BACKBONE_NAME = "dinov3_vitl16"
EMBEDDING_DIM = 1024
NUM_OUTPUTS = 3  # predicts [Dry_Clover_g, Dry_Dead_g, Dry_Green_g]

# MLP head settings (must match training)
HEAD_HIDDEN_DIMS = [1024, 512, 256]
HEAD_ACTIVATION = "relu"  # one of: relu, gelu, silu
HEAD_DROPOUT = 0.2
USE_OUTPUT_SOFTPLUS = True  # keep predictions non-negative

# Preprocessing
IMAGE_SIZE = 640
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TARGET_BASES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]

# Inference loader
BATCH_SIZE = 32
NUM_WORKERS = 4
# ==============================================


import os
import sys
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd

from torch import nn


# ===== Add local dinov3 to import path (strictly offline) =====
_DINOV3_DIR = os.path.abspath(DINOV3_PATH) if DINOV3_PATH else ""
if _DINOV3_DIR and os.path.isdir(_DINOV3_DIR) and _DINOV3_DIR not in sys.path:
    sys.path.insert(0, _DINOV3_DIR)

try:
    from dinov3.hub.backbones import dinov3_vitl16 as _dinov3_vitl16
except Exception:
    _dinov3_vitl16 = None


# ===== Offline-only weights loader (TorchScript supported, state_dict also supported) =====
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


def build_transforms() -> T.Compose:
    return T.Compose([
        T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD),
    ])


def load_state_and_meta(pt_path: str):
    # Deprecated: kept for backward compatibility with older code paths
    model, state_dict, meta = load_model_or_state(pt_path)
    if model is not None:
        return model, meta
    return state_dict, meta


class DinoV3FeatureExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Ensure dtype alignment (e.g., avoid Half vs Float mismatch)
        try:
            backbone_dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            backbone_dtype = images.dtype
        if images.dtype != backbone_dtype:
            images = images.to(dtype=backbone_dtype)
        feats = self.backbone.forward_features(images)
        return feats["x_norm_clstoken"]


def build_feature_extractor(backbone_name: str) -> nn.Module:
    if backbone_name != "dinov3_vitl16":
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    if _dinov3_vitl16 is None:
        raise ImportError(
            "dinov3 is not available locally. Ensure third_party/dinov3 is present and importable."
        )
    backbone = _dinov3_vitl16(pretrained=False)  # strictly offline, no downloads
    return DinoV3FeatureExtractor(backbone)


class BiomassRegressor(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_outputs: int = 3,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        head_dropout: float = 0.0,
        use_output_softplus: bool = True,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        feature_extractor = build_feature_extractor(backbone_name)
        if not freeze_backbone:
            for p in feature_extractor.backbone.parameters():
                p.requires_grad = True
            feature_extractor.backbone.train()
        self.feature_extractor = feature_extractor

        def build_activation(name: str) -> nn.Module:
            name = (name or "").lower()
            if name == "relu":
                return nn.ReLU(inplace=True)
            if name == "gelu":
                return nn.GELU()
            if name == "silu" or name == "swish":
                return nn.SiLU(inplace=True)
            return nn.ReLU(inplace=True)

        hidden_dims: List[int] = list(head_hidden_dims or [])
        layers: List[nn.Module] = []
        in_dim = embedding_dim
        act = build_activation(head_activation)

        if head_dropout and head_dropout > 0:
            layers.append(nn.Dropout(head_dropout))

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act.__class__())
            if head_dropout and head_dropout > 0:
                layers.append(nn.Dropout(head_dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_outputs))
        if use_output_softplus:
            layers.append(nn.Softplus())
        self.head = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        return self.head(features)


def build_model_from_meta(meta: Dict) -> BiomassRegressor:
    # Use meta for backbone/embedding/outputs if present, otherwise fallback to config
    backbone_name = str(meta.get("backbone", BACKBONE_NAME))
    embedding_dim = int(meta.get("embedding_dim", EMBEDDING_DIM))
    num_outputs = int(meta.get("num_outputs", NUM_OUTPUTS))
    model = BiomassRegressor(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        num_outputs=num_outputs,
        head_hidden_dims=HEAD_HIDDEN_DIMS,
        head_activation=HEAD_ACTIVATION,
        head_dropout=HEAD_DROPOUT,
        use_output_softplus=USE_OUTPUT_SOFTPLUS,
        freeze_backbone=True,
    )
    return model


def predict_for_images(model: nn.Module, dataset_root: str, image_paths: List[str], batch_size: int = BATCH_SIZE) -> Dict[str, Tuple[float, float, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only inference is enforced. CUDA is not available.")
    device = "cuda"

    tf = build_transforms()
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model.eval().to(device)
    torch.set_float32_matmul_precision("high")

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
    dataset_root, test_csv = resolve_paths(INPUT_PATH)
    df = pd.read_csv(test_csv)
    if not {"sample_id", "image_path", "target_name"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: sample_id, image_path, target_name")

    unique_image_paths = df["image_path"].astype(str).unique().tolist()

    model_loaded, state_dict = None, None
    model_or_state, meta = load_state_and_meta(WEIGHTS_PT_PATH)
    if isinstance(model_or_state, nn.Module):
        # TorchScript path
        model = model_or_state
    else:
        # state_dict path from export_ckpt_to_pt.py (offline, no downloads)
        state_dict = model_or_state
        model = build_model_from_meta(meta)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing}")
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected}")

    image_to_base_preds = predict_for_images(model, dataset_root, unique_image_paths, batch_size=32)

    # Build submission
    rows = []
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])  # e.g., IDxxxx__Dry_Clover_g
        rel_path = str(r["image_path"])  # e.g., test/IDxxxx.jpg
        target_name = str(r["target_name"])  # one of 5
        dc, dd, dg = image_to_base_preds.get(rel_path, (0.0, 0.0, 0.0))
        if target_name == "Dry_Clover_g":
            value = dc
        elif target_name == "Dry_Dead_g":
            value = dd
        elif target_name == "Dry_Green_g":
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


