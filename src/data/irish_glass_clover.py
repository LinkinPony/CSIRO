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
    - Only Dry_Total_g and Dry_Clover_g are supervised for reg3.
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
        reg3_mean: Optional[Sequence[float]] = None,
        reg3_std: Optional[Sequence[float]] = None,
        log_scale_targets: bool = False,
        transform: Optional[T.Compose] = None,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.root_dir = os.path.abspath(root_dir)
        self.image_subdir = image_subdir
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

        # Build reg3 targets in the global target_order, with a supervision mask
        reg3_vals: List[float] = []
        reg3_mask_vals: List[float] = []
        for name in self.target_order:
            if name == "Dry_Total_g":
                v = float(row["dry_total_g_cm2"])
                m = 1.0
            elif name == "Dry_Clover_g":
                v = float(row["dry_clover_g_cm2"])
                m = 1.0
            else:
                # No supervision for Dry_Green_g (and any other reg3 targets)
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

        return {
            "image": image,
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
            # Identifier used by MIR tiling logic to select per-dataset tile params.
            "dataset_id": "irish_glass_clover",
        }



