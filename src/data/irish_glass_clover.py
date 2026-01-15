from __future__ import annotations

import os
from typing import Optional, Sequence, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class IrishGlassCloverDataset(Dataset):
    """
    Dataset for the irish_glass_clover data.

    - Reads data.csv with columns including:
        image_name, dry_total_g_cm2, dry_clover_g_cm2, Sward Height (cm), ...
    - By default, only Dry_Total_g is supervised for reg3.
    - Optionally, this dataset can also supervise the biomass ratio head by mapping:
        - dry_clover_g_cm2                     -> Dry_Clover_g
        - dry_total_g_cm2 - dry_clover_g_cm2   -> Dry_Green_g
        - Dry_Dead_g                           = 0
    - Values in dry_total_g_cm2 / dry_clover_g_cm2 are already area-normalized
      (g per unit area, effectively g/m^2 for our purposes) and are used
      directly without any additional area conversion.
    - NDVI is NOT supervised for this dataset.
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        image_subdir: str,
        target_order: Sequence[str],
        supervise_ratio: bool = False,
        reg3_mean: Optional[Sequence[float]] = None,
        reg3_std: Optional[Sequence[float]] = None,
        log_scale_targets: bool = False,
        transform: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root_dir = os.path.abspath(root_dir)
        self.image_subdir = image_subdir
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

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        img_name = str(row["image_name"])
        img_path = os.path.join(self.root_dir, self.image_subdir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # Traceability/debug: provide an image_id compatible with the main pipeline.
        try:
            image_id = os.path.splitext(os.path.basename(img_name))[0]
        except Exception:
            image_id = str(img_name)

        # Build reg3 targets in the global target_order, with a supervision mask
        reg3_vals: List[float] = []
        reg3_mask_vals: List[float] = []
        for name in self.target_order:
            if name == "Dry_Total_g":
                v = float(row["dry_total_g_cm2"])
                m = 1.0
            else:
                # No supervision for any other reg3 targets for this dataset
                v = 0.0
                m = 0.0
            reg3_vals.append(v)
            reg3_mask_vals.append(m)

        y_reg3_g_m2 = torch.tensor(reg3_vals, dtype=torch.float32)
        reg3_mask = torch.tensor(reg3_mask_vals, dtype=torch.float32)

        # Apply optional log-scale + z-score normalization in-place
        y_reg3_norm = y_reg3_g_m2.clone()
        if self._log_scale_targets:
            y_reg3_norm = torch.log1p(torch.clamp(y_reg3_norm, min=0.0))
        if self._reg3_mean is not None and self._reg3_std is not None:
            safe_std = torch.clamp(self._reg3_std, min=1e-8)
            y_reg3_norm = (y_reg3_norm - self._reg3_mean) / safe_std

        # Height supervision is available via "Sward Height (cm)"
        try:
            h_raw = row.get("Sward Height (cm)", 0.0)
            y_height = torch.tensor(
                [float(h_raw) if h_raw == h_raw else 0.0], dtype=torch.float32
            )
        except Exception:
            y_height = torch.tensor([0.0], dtype=torch.float32)

        # No NDVI supervision for this dataset
        y_ndvi = torch.tensor([0.0], dtype=torch.float32)
        ndvi_mask = torch.tensor([0.0], dtype=torch.float32)

        # Species/state not available; keep as dummy zeros
        y_species = torch.tensor(0, dtype=torch.long)
        y_state = torch.tensor(0, dtype=torch.long)
        y_date = torch.zeros((2,), dtype=torch.float32)
        date_mask = torch.zeros((1,), dtype=torch.float32)

        # Biomass decomposition targets are not available in full for this dataset.
        # We always return dummy 5D tensors with zero masks so the collate path
        # stays consistent when mixing datasets.
        y_5d_g = torch.zeros(5, dtype=torch.float32)
        biomass_5d_mask = torch.zeros(5, dtype=torch.float32)

        # Optional ratio supervision (controls the ratio head only; does NOT enable 5D loss)
        y_ratio = torch.zeros(3, dtype=torch.float32)
        ratio_mask = torch.zeros(1, dtype=torch.float32)
        if self.supervise_ratio:
            try:
                total = float(row["dry_total_g_cm2"])
                clover = float(row["dry_clover_g_cm2"])
            except Exception:
                total = float("nan")
                clover = float("nan")
            if total == total and clover == clover and total > 0.0:
                dead = 0.0
                green = max(0.0, total - clover)
                y_ratio[0] = float(clover / total)
                y_ratio[1] = float(dead / total)  # always 0
                y_ratio[2] = float(green / total)
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
            # Biomass decomposition (unused for Irish; masks are zero)
            "y_biomass_5d_g": y_5d_g,
            "biomass_5d_mask": biomass_5d_mask,
            "y_ratio": y_ratio,
            "ratio_mask": ratio_mask,
        }


