from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F


@dataclass
class NdviDenseConfig:
    root: str
    tile_size: int = 512
    stride: int = 448
    batch_size: int = 1
    num_workers: int = 4
    mean: Sequence[float] = (0.485, 0.456, 0.406)
    std: Sequence[float] = (0.229, 0.224, 0.225)
    hflip_prob: float = 0.5
    vflip_prob: float = 0.0
    # Optional resize after crop (set to None to keep tile_size)
    resize: Optional[Tuple[int, int]] = None  # (h, w)


def _is_png(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in (".png",)


def _find_unique_by_substring(paths: List[str], substrings: Sequence[str]) -> Optional[str]:
    for s in substrings:
        cand = [p for p in paths if (s in os.path.basename(p).lower())]
        if len(cand) == 1:
            return cand[0]
        if len(cand) > 1:
            # Prefer the shortest filename to avoid picking *_cropped variants incorrectly when both exist
            cand.sort(key=lambda x: len(os.path.basename(x)))
            return cand[0]
    return None


def _scan_pairs(root: str) -> List[Tuple[str, str]]:
    """
    Scan root directory for pair folders of the form:
      <root>/{carrot,onion}/{1,2,...}/ containing *rgbreg*.png (RGB) and *ndvi*.png (label)
    Returns a list of (rgb_path, ndvi_path).
    """
    pairs: List[Tuple[str, str]] = []
    if not os.path.isdir(root):
        return pairs
    for cls in ("carrot", "onion"):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for name in sorted(os.listdir(cls_dir), key=lambda x: (len(x), x)):
            sub = os.path.join(cls_dir, name)
            if not os.path.isdir(sub):
                continue
            # Collect PNGs in the subdir
            pngs = [os.path.join(sub, f) for f in os.listdir(sub) if _is_png(os.path.join(sub, f))]
            lower = [os.path.basename(p).lower() for p in pngs]
            # Find RGBREG and NDVI candidates
            rgb_path = _find_unique_by_substring(pngs, substrings=("rgbreg",))
            ndvi_path = _find_unique_by_substring(pngs, substrings=("ndvi",))
            if rgb_path is None or ndvi_path is None:
                continue
            pairs.append((rgb_path, ndvi_path))
    return pairs


def _compute_tiles(w: int, h: int, tile: int, stride: int) -> List[Tuple[int, int, int, int]]:
    boxes: List[Tuple[int, int, int, int]] = []
    if tile <= 0:
        return [(0, 0, w, h)]
    x_starts = list(range(0, max(1, w - tile + 1), max(1, stride)))
    y_starts = list(range(0, max(1, h - tile + 1), max(1, stride)))
    if len(x_starts) == 0:
        x_starts = [0]
    if len(y_starts) == 0:
        y_starts = [0]
    # Ensure coverage of right/bottom edges
    if x_starts[-1] != max(0, w - tile):
        x_starts.append(max(0, w - tile))
    if y_starts[-1] != max(0, h - tile):
        y_starts.append(max(0, h - tile))
    for y in y_starts:
        for x in x_starts:
            x2 = min(x + tile, w)
            y2 = min(y + tile, h)
            x1 = max(0, x2 - tile)
            y1 = max(0, y2 - tile)
            boxes.append((x1, y1, x2, y2))
    return boxes


def _to_ndvi_float(arr: np.ndarray) -> np.ndarray:
    """
    Map NDVI PNG to [-1, 1].
    - If uint8: x in [0,255] -> x/127.5 - 1
    - If uint16: x in [0,65535] -> (x/65535)*2 - 1
    - If float: clip to [-1,1]
    """
    if arr.dtype == np.uint8:
        out = (arr.astype(np.float32) / 127.5) - 1.0
        return np.clip(out, -1.0, 1.0)
    if arr.dtype == np.uint16:
        out = (arr.astype(np.float32) / 65535.0) * 2.0 - 1.0
        return np.clip(out, -1.0, 1.0)
    out = arr.astype(np.float32)
    return np.clip(out, -1.0, 1.0)


class NdviDenseTilesDataset(Dataset):
    def __init__(
        self,
        cfg: NdviDenseConfig,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = str(split).lower()
        # Build per-tile index across all image pairs
        self._pairs: List[Tuple[str, str]] = _scan_pairs(cfg.root)
        self._index: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for idx, (rgb_p, ndvi_p) in enumerate(self._pairs):
            try:
                with Image.open(rgb_p) as img:
                    w, h = img.size
            except Exception:
                # fallback to ndvi for size
                with Image.open(ndvi_p) as img:
                    w, h = img.size
            boxes = _compute_tiles(w, h, cfg.tile_size, cfg.stride)
            for box in boxes:
                self._index.append((idx, box))
        # Transforms
        self._normalize = T.Normalize(mean=list(cfg.mean), std=list(cfg.std))
        self._train_hflip_p = float(cfg.hflip_prob)
        self._train_vflip_p = float(cfg.vflip_prob)
        self._resize_size: Optional[Tuple[int, int]] = tuple(cfg.resize) if (cfg.resize is not None and len(cfg.resize) == 2) else None

    def __len__(self) -> int:
        return len(self._index)

    def _maybe_flip(self, img: Image.Image, mask: Optional[Image.Image] = None) -> Tuple[Image.Image, Optional[Image.Image], Dict[str, Any]]:
        ops: Dict[str, Any] = {}
        if self.split == "train":
            if self._train_hflip_p > 0.0 and torch.rand(()) < self._train_hflip_p:
                img = F.hflip(img)
                mask = F.hflip(mask) if mask is not None else None
                ops["hflip"] = True
            if self._train_vflip_p > 0.0 and torch.rand(()) < self._train_vflip_p:
                img = F.vflip(img)
                mask = F.vflip(mask) if mask is not None else None
                ops["vflip"] = True
        return img, mask, ops

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair_idx, (x1, y1, x2, y2) = self._index[index]
        rgb_path, ndvi_path = self._pairs[pair_idx]
        with Image.open(rgb_path) as im_rgb:
            rgb = im_rgb.convert("RGB").crop((x1, y1, x2, y2))
        with Image.open(ndvi_path) as im_ndvi:
            ndvi = im_ndvi.convert("L").crop((x1, y1, x2, y2))

        # Optional resize after crop (for non-square or different scale)
        if self._resize_size is not None:
            rgb = F.resize(rgb, self._resize_size, interpolation=T.InterpolationMode.BILINEAR)
            ndvi = F.resize(ndvi, self._resize_size, interpolation=T.InterpolationMode.BILINEAR)

        # Paired flips
        rgb, ndvi, _ = self._maybe_flip(rgb, ndvi)

        # To tensor and normalize image
        rgb_t = F.to_tensor(rgb)
        rgb_t = self._normalize(rgb_t)

        # NDVI: to float in [-1,1]
        ndvi_np = np.array(ndvi)
        ndvi_f = _to_ndvi_float(ndvi_np)
        ndvi_t = torch.from_numpy(ndvi_f).unsqueeze(0)  # 1 x H x W
        valid_mask = torch.isfinite(ndvi_t) & (ndvi_t >= -1.0) & (ndvi_t <= 1.0)

        return {
            "image": rgb_t,
            "ndvi_dense": ndvi_t.to(torch.float32),
            "ndvi_mask": valid_mask.to(torch.bool),
            "task": "ndvi_dense",
            "meta": {
                "rgb_path": rgb_path,
                "ndvi_path": ndvi_path,
                "box": (x1, y1, x2, y2),
            },
        }


def build_ndvi_dense_dataloader(cfg: NdviDenseConfig, split: str = "train") -> DataLoader:
    ds = NdviDenseTilesDataset(cfg=cfg, split=split)
    is_train = str(split).lower() == "train"
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=is_train,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.num_workers > 0),
        drop_last=is_train,
    )



class NdviDenseAsScalarDataset(Dataset):
    """
    NDVI-dense tiles dataset that converts the dense NDVI map to a scalar label
    by averaging valid pixels, and applies the same image transforms as the main
    regression dataset.
    """
    def __init__(
        self,
        cfg: NdviDenseConfig,
        split: str = "train",
        transform: Optional[T.Compose] = None,
        reg3_dim: int = 3,
        ndvi_mean: Optional[float] = None,
        ndvi_std: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.split = str(split).lower()
        self.transform = transform
        self.reg3_dim = int(reg3_dim)
        self._ndvi_mean = float(ndvi_mean) if ndvi_mean is not None else None
        self._ndvi_std = float(ndvi_std) if ndvi_std is not None else None
        # Build index over tiles
        self._pairs: List[Tuple[str, str]] = _scan_pairs(cfg.root)
        self._index: List[Tuple[int, Tuple[int, int, int, int]]] = []
        for idx, (rgb_p, ndvi_p) in enumerate(self._pairs):
            try:
                with Image.open(rgb_p) as img:
                    w, h = img.size
            except Exception:
                with Image.open(ndvi_p) as img:
                    w, h = img.size
            boxes = _compute_tiles(w, h, cfg.tile_size, cfg.stride)
            for box in boxes:
                self._index.append((idx, box))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair_idx, (x1, y1, x2, y2) = self._index[index]
        rgb_path, ndvi_path = self._pairs[pair_idx]
        with Image.open(rgb_path) as im_rgb:
            rgb = im_rgb.convert("RGB").crop((x1, y1, x2, y2))
        with Image.open(ndvi_path) as im_ndvi:
            ndvi = im_ndvi.convert("L").crop((x1, y1, x2, y2))

        # Compute scalar NDVI label from dense map (mean over valid pixels)
        ndvi_np = np.array(ndvi)
        ndvi_f = _to_ndvi_float(ndvi_np)
        ndvi_t = torch.from_numpy(ndvi_f).unsqueeze(0)  # 1 x H x W
        valid_mask = torch.isfinite(ndvi_t) & (ndvi_t >= -1.0) & (ndvi_t <= 1.0)
        denom = valid_mask.sum().clamp_min(1)
        y_ndvi_scalar = (ndvi_t[valid_mask].sum() / denom).to(torch.float32).unsqueeze(0)  # (1,)
        # Apply z-score if provided
        if self._ndvi_mean is not None and self._ndvi_std is not None:
            ndvi_std = max(1e-8, float(self._ndvi_std))
            y_ndvi_scalar = (y_ndvi_scalar - float(self._ndvi_mean)) / ndvi_std

        # Apply main dataset transforms to image
        if self.transform is not None:
            rgb = self.transform(rgb)
        else:
            rgb = T.ToTensor()(rgb)

        # Placeholder targets to satisfy CutMix/main collate
        y_reg3 = torch.zeros((self.reg3_dim,), dtype=torch.float32)
        y_height = torch.zeros((1,), dtype=torch.float32)
        y_species = torch.tensor(0, dtype=torch.long)
        y_state = torch.tensor(0, dtype=torch.long)

        return {
            "image": rgb,
            "y_reg3": y_reg3,
            "y_height": y_height,
            "y_ndvi": y_ndvi_scalar,
            "y_species": y_species,
            "y_state": y_state,
            "ndvi_only": True,
            "task": "ndvi_only",
            "meta": {
                "rgb_path": rgb_path,
                "ndvi_path": ndvi_path,
                "box": (x1, y1, x2, y2),
            },
        }


def build_ndvi_scalar_dataloader(
    cfg: NdviDenseConfig,
    split: str,
    transform: Optional[T.Compose],
    reg3_dim: int = 3,
    ndvi_mean: Optional[float] = None,
    ndvi_std: Optional[float] = None,
) -> DataLoader:
    ds = NdviDenseAsScalarDataset(cfg=cfg, split=split, transform=transform, reg3_dim=reg3_dim, ndvi_mean=ndvi_mean, ndvi_std=ndvi_std)
    is_train = str(split).lower() == "train"
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=is_train,
        num_workers=int(cfg.num_workers),
        pin_memory=True,
        persistent_workers=bool(cfg.num_workers > 0),
        drop_last=is_train,
    )


