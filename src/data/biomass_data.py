from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class BiomassDataDataset(Dataset):
    """
    Dataset for the biomass_data (train/test) CSVs.

    Expected CSV columns (semicolon-separated):
      - image_file_name
      - dry_total, dry_clover, dry_weeds, dry_grass

    Supervision mirrors the Irish dataset:
      - reg3: only Dry_Total_g (others masked out)
      - ratio: derived from dry_clover, dry_weeds + dry_grass; Dry_Dead_g = 0

    Values are treated as area-normalized (g/m^2) and used directly.
    """

    def __init__(
        self,
        csv_paths: Sequence[str],
        root_dir: str,
        target_order: Sequence[str],
        image_dir: str = "images",
        image_dir_from_csv: bool = True,
        image_col: str = "image_file_name",
        dry_total_col: str = "dry_total",
        dry_clover_col: str = "dry_clover",
        dry_weeds_col: str = "dry_weeds",
        dry_grass_col: str = "dry_grass",
        supervise_ratio: bool = False,
        drop_unlabeled: bool = False,
        reg3_mean: Optional[Sequence[float]] = None,
        reg3_std: Optional[Sequence[float]] = None,
        log_scale_targets: bool = False,
        transform: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        self.root_dir = os.path.abspath(root_dir)
        self.image_dir = str(image_dir or "")
        self.image_dir_from_csv = bool(image_dir_from_csv)
        self.image_col = str(image_col or "image_file_name")
        self.dry_total_col = str(dry_total_col or "dry_total")
        self.dry_clover_col = str(dry_clover_col or "dry_clover")
        self.dry_weeds_col = str(dry_weeds_col or "dry_weeds")
        self.dry_grass_col = str(dry_grass_col or "dry_grass")
        self.supervise_ratio: bool = bool(supervise_ratio)
        self.target_order: List[str] = list(target_order)
        self.transform = transform

        # Normalization (shared with CSIRO pipeline)
        self._log_scale_targets: bool = bool(log_scale_targets)
        self._reg3_mean = (
            torch.tensor(list(reg3_mean), dtype=torch.float32)
            if reg3_mean is not None and len(reg3_mean) == len(self.target_order)
            else None
        )
        self._reg3_std = (
            torch.tensor(list(reg3_std), dtype=torch.float32)
            if reg3_std is not None and len(reg3_std) == len(self.target_order)
            else None
        )

        dfs: List[pd.DataFrame] = []
        for csv_path in csv_paths:
            if not csv_path:
                continue
            df = pd.read_csv(str(csv_path), sep=";", encoding="utf-8-sig")
            df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
            if self.image_col not in df.columns:
                for alt in ("image_file_name", "image_name", "image", "filename", "file"):
                    if alt in df.columns:
                        self.image_col = alt
                        break
            if self.image_col not in df.columns:
                raise ValueError(f"Missing image column '{self.image_col}' in {csv_path}")
            if "image_path" not in df.columns:
                base_dir = ""
                if self.image_dir_from_csv:
                    split_name = os.path.basename(os.path.dirname(str(csv_path)))
                    if self.image_dir:
                        base_dir = os.path.join(split_name, self.image_dir)
                    else:
                        base_dir = split_name
                else:
                    base_dir = self.image_dir
                df = df.copy()
                df["image_path"] = df[self.image_col].map(
                    lambda x: os.path.join(base_dir, str(x)) if base_dir else str(x)
                )
            dfs.append(df)

        if dfs:
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.DataFrame()

        for col in (self.dry_total_col, self.dry_clover_col, self.dry_weeds_col, self.dry_grass_col):
            if col not in self.df.columns:
                self.df[col] = float("nan")

        if drop_unlabeled and len(self.df) > 0:
            totals = pd.to_numeric(self.df[self.dry_total_col], errors="coerce")
            self.df = self.df[totals.notna()].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _safe_float(self, row: pd.Series, col: str) -> float:
        try:
            v = float(row.get(col, float("nan")))
        except Exception:
            v = float("nan")
        return v

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        rel_path = str(row.get("image_path", "")).strip()
        if os.path.isabs(rel_path):
            img_path = rel_path
        else:
            img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        try:
            image_id = os.path.splitext(os.path.basename(rel_path))[0]
        except Exception:
            image_id = str(rel_path)

        total = self._safe_float(row, self.dry_total_col)
        clover = self._safe_float(row, self.dry_clover_col)
        weeds = self._safe_float(row, self.dry_weeds_col)
        grass = self._safe_float(row, self.dry_grass_col)

        total_ok = math.isfinite(total)
        total_val = float(total) if total_ok else 0.0

        # Build reg3 targets in the global target_order, with a supervision mask
        reg3_vals: List[float] = []
        reg3_mask_vals: List[float] = []
        for name in self.target_order:
            if name == "Dry_Total_g":
                reg3_vals.append(total_val)
                reg3_mask_vals.append(1.0 if total_ok else 0.0)
            else:
                reg3_vals.append(0.0)
                reg3_mask_vals.append(0.0)

        y_reg3_g_m2 = torch.tensor(reg3_vals, dtype=torch.float32)
        reg3_mask = torch.tensor(reg3_mask_vals, dtype=torch.float32)

        # Apply optional log-scale + z-score normalization in-place
        y_reg3_norm = y_reg3_g_m2.clone()
        if self._log_scale_targets:
            y_reg3_norm = torch.log1p(torch.clamp(y_reg3_norm, min=0.0))
        if self._reg3_mean is not None and self._reg3_std is not None:
            safe_std = torch.clamp(self._reg3_std, min=1e-8)
            y_reg3_norm = (y_reg3_norm - self._reg3_mean) / safe_std

        # No NDVI supervision for this dataset
        y_ndvi = torch.tensor([0.0], dtype=torch.float32)
        ndvi_mask = torch.tensor([0.0], dtype=torch.float32)

        # Species/state/height not available; keep as dummy zeros
        y_height = torch.tensor([0.0], dtype=torch.float32)
        y_species = torch.tensor(0, dtype=torch.long)
        y_state = torch.tensor(0, dtype=torch.long)
        y_date = torch.zeros((2,), dtype=torch.float32)
        date_mask = torch.zeros((1,), dtype=torch.float32)

        # Biomass decomposition targets are not available in full for this dataset.
        y_5d_g = torch.zeros(5, dtype=torch.float32)
        biomass_5d_mask = torch.zeros(5, dtype=torch.float32)

        # Optional ratio supervision (controls the ratio head only; does NOT enable 5D loss)
        y_ratio = torch.zeros(3, dtype=torch.float32)
        ratio_mask = torch.zeros(1, dtype=torch.float32)
        if self.supervise_ratio:
            clover_ok = math.isfinite(clover)
            weeds_ok = math.isfinite(weeds)
            grass_ok = math.isfinite(grass)
            if total_ok and total > 0.0 and clover_ok and weeds_ok and grass_ok:
                green = max(0.0, float(weeds) + float(grass))
                y_ratio[0] = float(clover / total)  # Dry_Clover_g
                y_ratio[1] = 0.0  # Dry_Dead_g
                y_ratio[2] = float(green / total)   # Dry_Green_g
                ratio_mask[...] = 1.0

        return {
            "image": image,
            "image_id": image_id,
            # main regression (normalized + original g/m^2)
            "y_reg3": y_reg3_norm,
            "y_reg3_g_m2": y_reg3_g_m2,
            "y_reg3_g": y_reg3_g_m2,  # treated as area-normalized; not used for submission
            "reg3_mask": reg3_mask,
            # auxiliary targets
            "y_height": y_height,
            "y_ndvi": y_ndvi,
            "ndvi_mask": ndvi_mask,
            "y_species": y_species,
            "y_state": y_state,
            "y_date": y_date,
            "date_mask": date_mask,
            # Biomass decomposition (unused; masks are zero)
            "y_biomass_5d_g": y_5d_g,
            "biomass_5d_mask": biomass_5d_mask,
            "y_ratio": y_ratio,
            "ratio_mask": ratio_mask,
        }
