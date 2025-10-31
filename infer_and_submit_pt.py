import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd

from src.models.regressor import BiomassRegressor


# ===== User variables (edit these three) =====
WEIGHTS_PT_PATH = "/home/dl/Git/CSIRO/weights/best.pt"  # exported by export_ckpt_to_pt.py
INPUT_PATH = "/home/dl/Git/CSIRO/data"  # dir containing test.csv & images, or a direct test.csv path
OUTPUT_SUBMISSION_PATH = "/home/dl/Git/CSIRO/submission.csv"
# ============================================

IMAGE_SIZE = 640
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TARGET_BASES = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]


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
    if not os.path.isfile(pt_path):
        raise FileNotFoundError(f"Weights not found: {pt_path}")
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj.get("meta", {})
    # raw state_dict
    return obj, {}


def build_model_from_meta(meta: Dict) -> BiomassRegressor:
    backbone_name = str(meta.get("backbone", "dinov3_vitl16"))
    embedding_dim = int(meta.get("embedding_dim", 1024))
    num_outputs = int(meta.get("num_outputs", 3))
    # Build without downloading weights; we'll load our state_dict next
    model = BiomassRegressor(
        backbone_name=backbone_name,
        embedding_dim=embedding_dim,
        num_outputs=num_outputs,
        dropout=0.0,
        pretrained=False,
        weights_url=None,
        freeze_backbone=True,
    )
    return model


def predict_for_images(model: BiomassRegressor, dataset_root: str, image_paths: List[str], batch_size: int = 32) -> Dict[str, Tuple[float, float, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU-only inference is enforced. CUDA is not available.")
    device = "cuda"

    tf = build_transforms()
    ds = TestImageDataset(image_paths, root_dir=dataset_root, transform=tf)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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

    state_dict, meta = load_state_and_meta(WEIGHTS_PT_PATH)
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


