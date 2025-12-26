import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule
from src.data.augmentations import build_transforms as build_aug_transforms
from .ndvi_dense import NdviDenseConfig, build_ndvi_scalar_dataloader
from .irish_glass_clover import IrishGlassCloverDataset


@dataclass
class NormalizationSpec:
    mean: Sequence[float]
    std: Sequence[float]


def _merge_augment_cfg(
    augment_cfg: Optional[Dict],
    train_scale: Tuple[float, float],
    hflip_prob: float,
) -> Dict:
    base = dict(augment_cfg or {})
    # Backward-compat keys used in existing configs
    if "random_resized_crop_scale" not in base:
        base["random_resized_crop_scale"] = tuple(train_scale)
    if "horizontal_flip_prob" not in base and "horizontal_flip" not in base:
        base["horizontal_flip_prob"] = float(hflip_prob)
    return base


class _GroupedConcatBatchSampler:
    """
    Batch sampler for `ConcatDataset` that yields batches containing samples from
    a *single* underlying sub-dataset.

    Motivation:
      - When manifold mixup is enabled, mixing patch tokens across datasets can
        be unstable and/or invalid (different image sizes -> different token grids).
      - Grouping by dataset ensures in-batch manifold mixup never crosses datasets.

    Notes:
      - We shuffle *within* each dataset and then shuffle the resulting list of
        batches to interleave datasets across steps while keeping batches homogeneous.
      - We try to avoid a final batch of size 1 (best-effort) by borrowing one
        sample from the previous batch of the same dataset when possible.
    """

    def __init__(self, ds: ConcatDataset, *, batch_size: int, shuffle: bool) -> None:
        if not isinstance(ds, ConcatDataset):
            raise TypeError("_GroupedConcatBatchSampler expects a ConcatDataset")
        bsz = int(batch_size)
        if bsz <= 0:
            raise ValueError("batch_size must be > 0")
        self.ds = ds
        self.batch_size = bsz
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        total = 0
        for sub in getattr(self.ds, "datasets", []):
            try:
                n = int(len(sub))
            except Exception:
                n = 0
            if n <= 0:
                continue
            full = n // self.batch_size
            rem = n % self.batch_size
            total += full + (1 if rem > 0 else 0)
        return int(total)

    def __iter__(self):
        # Build per-dataset index lists (global indices in ConcatDataset).
        all_batches: List[List[int]] = []
        offset = 0
        for sub in getattr(self.ds, "datasets", []):
            n = int(len(sub))
            if n <= 0:
                continue
            # Local indices [0..n), then add offset to become ConcatDataset indices.
            if self.shuffle and n > 1:
                perm = torch.randperm(n).tolist()
            else:
                perm = list(range(n))
            idxs = [offset + i for i in perm]

            # Chunk into batches.
            batches: List[List[int]] = []
            for i in range(0, n, self.batch_size):
                chunk = idxs[i : i + self.batch_size]
                if chunk:
                    batches.append(chunk)

            # Best-effort: avoid a last batch of size 1 by borrowing from previous.
            if (
                len(batches) >= 2
                and len(batches[-1]) == 1
                and len(batches[-2]) >= 3  # ensures previous does not become size-1
            ):
                last = batches[-1]
                prev = batches[-2]
                last.insert(0, prev.pop())

            all_batches.extend(batches)
            offset += n

        # Shuffle batch order to interleave datasets while keeping each batch homogeneous.
        if self.shuffle and len(all_batches) > 1:
            order = torch.randperm(len(all_batches)).tolist()
            for j in order:
                yield all_batches[j]
        else:
            for b in all_batches:
                yield b


class PastureImageDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        root_dir: str,
        target_order: Sequence[str],
        area_m2: float = 1.0,
        reg3_mean: Optional[Sequence[float]] = None,
        reg3_std: Optional[Sequence[float]] = None,
        ndvi_mean: Optional[float] = None,
        ndvi_std: Optional[float] = None,
        log_scale_targets: bool = False,
        species_to_idx: Optional[dict] = None,
        state_to_idx: Optional[dict] = None,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.root_dir = root_dir
        self.target_order = list(target_order)
        self.area_m2 = float(area_m2) if area_m2 is not None else 1.0
        # z-score params
        self._reg3_mean = torch.tensor(list(reg3_mean), dtype=torch.float32) if reg3_mean is not None else None
        self._reg3_std = torch.tensor(list(reg3_std), dtype=torch.float32) if reg3_std is not None else None
        self._ndvi_mean = float(ndvi_mean) if ndvi_mean is not None else None
        self._ndvi_std = float(ndvi_std) if ndvi_std is not None else None
        self._log_scale_targets = bool(log_scale_targets)
        self.species_to_idx = dict(species_to_idx or {})
        self.state_to_idx = dict(state_to_idx or {})
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image_path = os.path.join(self.root_dir, str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # For debugging / traceability (e.g., dumping augmented inputs), keep the original image id.
        # This is the CSV-derived id without the target suffix (e.g. "ID123456789").
        try:
            image_id = str(row.get("image_id", "")).strip()
        except Exception:
            image_id = ""
        if not image_id:
            try:
                # Fallback to the file stem (works for typical "IDxxxx.jpg" paths).
                image_id = os.path.splitext(os.path.basename(str(row.get("image_path", ""))))[0]
            except Exception:
                image_id = ""
        # main regression targets (one or more components depending on config.target_order)
        y_reg3_g = torch.tensor([row[t] for t in self.target_order], dtype=torch.float32)
        reg3_mask = torch.ones_like(y_reg3_g, dtype=torch.float32)
        # Convert grams to grams per square meter if a valid area is provided
        y_reg3_g_m2 = y_reg3_g / float(self.area_m2) if self.area_m2 > 0.0 else y_reg3_g
        
        # Apply optional log-scale + z-score normalization
        y_reg3_norm = y_reg3_g_m2.clone()
        if self._log_scale_targets:
            y_reg3_norm = torch.log1p(torch.clamp(y_reg3_norm, min=0.0))

        # Apply z-score if provided
        if self._reg3_mean is not None and self._reg3_std is not None:
            safe_std = torch.clamp(self._reg3_std, min=1e-8)
            y_reg3 = (y_reg3_norm - self._reg3_mean) / safe_std
        else:
            y_reg3 = y_reg3_norm
        # --- Canonical biomass components in grams (CSIRO only) ---
        # Order: [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
        try:
            dry_clover = float(row.get("Dry_Clover_g", float("nan")))
            dry_dead = float(row.get("Dry_Dead_g", float("nan")))
            dry_green = float(row.get("Dry_Green_g", float("nan")))
            gdm = float(row.get("GDM_g", float("nan")))
            dry_total = float(row.get("Dry_Total_g", float("nan")))
        except Exception:
            dry_clover = dry_dead = dry_green = gdm = dry_total = float("nan")
        y_5d_g = torch.tensor(
            [dry_clover, dry_dead, dry_green, gdm, dry_total],
            dtype=torch.float32,
        )
        biomass_5d_mask = torch.isfinite(y_5d_g).to(dtype=torch.float32)

        # Ratio targets: proportions of (Dry_Clover_g, Dry_Dead_g, Dry_Green_g) over Dry_Total_g.
        # Only valid for CSIRO samples where all components are present and Dry_Total_g > 0.
        y_ratio = torch.zeros(3, dtype=torch.float32)
        ratio_mask = torch.zeros(1, dtype=torch.float32)
        if torch.isfinite(y_5d_g[-1]) and y_5d_g[-1] > 0:
            total = y_5d_g[-1]
            clover = y_5d_g[0] if torch.isfinite(y_5d_g[0]) else 0.0
            dead = y_5d_g[1] if torch.isfinite(y_5d_g[1]) else 0.0
            green = y_5d_g[2] if torch.isfinite(y_5d_g[2]) else 0.0
            y_ratio[0] = clover / total
            y_ratio[1] = dead / total
            y_ratio[2] = green / total
            ratio_mask[...] = 1.0

        # auxiliary regression target: height
        y_height = torch.tensor([float(row.get("Height_Ave_cm", 0.0))], dtype=torch.float32)
        # auxiliary regression target: Pre_GSHH_NDVI
        y_ndvi_raw = torch.tensor([float(row.get("Pre_GSHH_NDVI", 0.0))], dtype=torch.float32)
        if self._ndvi_mean is not None and self._ndvi_std is not None:
            ndvi_std = max(1e-8, float(self._ndvi_std))
            y_ndvi = (y_ndvi_raw - float(self._ndvi_mean)) / ndvi_std
        else:
            y_ndvi = y_ndvi_raw
        ndvi_mask = torch.ones((1,), dtype=torch.float32)
        # auxiliary classification target: species index
        species = str(row.get("Species", ""))
        if self.species_to_idx and species in self.species_to_idx:
            y_species = torch.tensor(int(self.species_to_idx[species]), dtype=torch.long)
        else:
            y_species = torch.tensor(0, dtype=torch.long)
        # auxiliary classification target: state index
        state = str(row.get("State", ""))
        if self.state_to_idx and state in self.state_to_idx:
            y_state = torch.tensor(int(self.state_to_idx[state]), dtype=torch.long)
        else:
            y_state = torch.tensor(0, dtype=torch.long)
        return {
            "image": image,
            "image_id": image_id,
            "y_reg3": y_reg3,             # normalized (if stats provided)
            "y_reg3_g_m2": y_reg3_g_m2,   # original g/m^2
            "y_reg3_g": y_reg3_g,         # original grams
            "reg3_mask": reg3_mask,       # all ones for CSIRO dataset
            "y_height": y_height,
            "y_ndvi": y_ndvi,             # normalized (if stats provided)
            "ndvi_mask": ndvi_mask,       # 1 => NDVI supervised
            "y_species": y_species,
            "y_state": y_state,
            # Biomass decomposition targets (CSIRO only; masks handle missing values)
            "y_biomass_5d_g": y_5d_g,
            "biomass_5d_mask": biomass_5d_mask,
            "y_ratio": y_ratio,
            "ratio_mask": ratio_mask,
        }


class PastureDataModule(LightningDataModule):
    def __init__(
        self,
        data_root: str,
        train_csv: str,
        image_size: Union[int, Tuple[int, int]],
        batch_size: int,
        val_batch_size: int,
        num_workers: int,
        val_split: float,
        target_order: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        train_scale: Tuple[float, float] = (0.8, 1.0),
        sample_area_m2: float = 1.0,
        zscore_output_path: Optional[str] = None,
        log_scale_targets: bool = False,
        hflip_prob: float = 0.5,
        augment_cfg: Optional[Dict] = None,
        shuffle: bool = True,
        prefetch_factor: int = 2,
        predefined_train_df: Optional[pd.DataFrame] = None,
        predefined_val_df: Optional[pd.DataFrame] = None,
        # Optional NDVI-dense settings
        ndvi_dense_enabled: bool = False,
        ndvi_dense_root: Optional[str] = None,
        ndvi_dense_tile_size: int = 512,
        ndvi_dense_stride: int = 448,
        ndvi_dense_batch_size: Optional[int] = None,
        ndvi_dense_num_workers: Optional[int] = None,
        ndvi_dense_mean: Optional[Sequence[float]] = None,
        ndvi_dense_std: Optional[Sequence[float]] = None,
        ndvi_dense_hflip_prob: float = 0.5,
        ndvi_dense_vflip_prob: float = 0.0,
        # Optional Irish Glass Clover mixed dataset
        irish_enabled: bool = False,
        irish_root: Optional[str] = None,
        irish_csv: Optional[str] = None,
        irish_image_dir: str = "images",
        irish_supervise_ratio: bool = False,
        irish_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        # Seed for reproducible internal train/val split when predefined splits are not supplied
        random_seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.train_csv = train_csv
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.prefetch_factor = int(prefetch_factor)
        self.val_split = float(val_split)
        self.target_order = list(target_order)
        self.shuffle = shuffle
        self.sample_area_m2 = float(sample_area_m2) if sample_area_m2 is not None else 1.0
        self._zscore_output_path: Optional[str] = str(zscore_output_path) if zscore_output_path else None
        self._log_scale_targets: bool = bool(log_scale_targets)
        merged_aug = _merge_augment_cfg(augment_cfg=augment_cfg, train_scale=train_scale, hflip_prob=hflip_prob)
        # Whether manifold mixup is enabled (used to decide batch composition when mixing datasets).
        try:
            mm = dict(merged_aug.get("manifold_mixup", {}) or {})
            self._manifold_mixup_enabled = bool(mm.get("enabled", False)) and float(mm.get("prob", 0.0)) > 0.0
        except Exception:
            self._manifold_mixup_enabled = False
        self.train_tf, self.val_tf = build_aug_transforms(
            image_size=image_size, mean=mean, std=std, augment_cfg=merged_aug
        )

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self._predefined_train_df: Optional[pd.DataFrame] = predefined_train_df
        self._predefined_val_df: Optional[pd.DataFrame] = predefined_val_df
        # NDVI-dense config (optional)
        self.ndvi_dense_enabled = bool(ndvi_dense_enabled)
        self._ndvi_cfg: Optional[NdviDenseConfig] = None
        if self.ndvi_dense_enabled and ndvi_dense_root:
            self._ndvi_cfg = NdviDenseConfig(
                root=str(ndvi_dense_root),
                tile_size=int(ndvi_dense_tile_size),
                stride=int(ndvi_dense_stride),
                batch_size=int(ndvi_dense_batch_size if ndvi_dense_batch_size is not None else batch_size),
                num_workers=int(ndvi_dense_num_workers if ndvi_dense_num_workers is not None else num_workers),
                mean=list(ndvi_dense_mean) if ndvi_dense_mean is not None else list(mean),
                std=list(ndvi_dense_std) if ndvi_dense_std is not None else list(std),
                hflip_prob=float(ndvi_dense_hflip_prob),
                vflip_prob=float(ndvi_dense_vflip_prob),
            )
        # Irish Glass Clover mixed dataset config
        self.irish_enabled: bool = bool(irish_enabled)
        self.irish_root: Optional[str] = str(irish_root) if irish_root else None
        self.irish_csv: Optional[str] = str(irish_csv) if irish_csv else None
        self.irish_image_dir: str = str(irish_image_dir or "images")
        self.irish_supervise_ratio: bool = bool(irish_supervise_ratio)
        self.irish_image_size: Optional[Union[int, Tuple[int, int]]] = irish_image_size
        try:
            self.random_seed: int = int(random_seed)
        except Exception:
            self.random_seed = 42
        self.irish_train_tf: Optional[T.Compose] = None
        if self.irish_enabled and self.irish_root and self.irish_csv:
            # Reuse the same augment cfg as CSIRO, but allow a different image_size
            irish_size = self.irish_image_size if self.irish_image_size is not None else image_size
            merged_aug_irish = _merge_augment_cfg(augment_cfg=augment_cfg, train_scale=train_scale, hflip_prob=hflip_prob)
            self.irish_train_tf, _ = build_aug_transforms(
                image_size=irish_size, mean=mean, std=std, augment_cfg=merged_aug_irish
            )
        # z-score stats (computed in setup)
        self._reg3_mean: Optional[List[float]] = None
        self._reg3_std: Optional[List[float]] = None
        self._ndvi_mean: Optional[float] = None
        self._ndvi_std: Optional[float] = None
        # Optional z-score stats for full 5D biomass components:
        # [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
        self._biomass_5d_mean: Optional[List[float]] = None
        self._biomass_5d_std: Optional[List[float]] = None

    def _read_and_pivot(self) -> pd.DataFrame:
        csv_path = os.path.join(self.data_root, self.train_csv)
        df = pd.read_csv(csv_path)
        # Do NOT filter by target_order here: we want to keep all canonical biomass
        # components (Dry_Green_g, Dry_Dead_g, Dry_Clover_g, GDM_g, Dry_Total_g)
        # so that auxiliary losses (ratio head, 5D weighted MSE) can be computed
        # even when the main regression head supervises only a subset (e.g., Dry_Total_g).
        # sample_id in CSV includes target suffix, e.g., IDxxxx__Dry_Clover_g
        # derive an image_id without suffix to aggregate targets per image
        df = df.copy()
        df["image_id"] = df["sample_id"].astype(str).str.split("__", n=1, expand=True)[0]

        pivot = df.pivot_table(
            index="image_id",
            columns="target_name",
            values="target",
            aggfunc="first",
        )
        image_path_series = df.groupby("image_id")["image_path"].first()
        # also aggregate auxiliary labels
        # Keep Sampling_Date at image-level so k-fold can optionally group by (Sampling_Date, State).
        sampling_date_series = df.groupby("image_id")["Sampling_Date"].first()
        height_series = df.groupby("image_id")["Height_Ave_cm"].first()
        ndvi_series = df.groupby("image_id")["Pre_GSHH_NDVI"].first()
        species_series = df.groupby("image_id")["Species"].first()
        state_series = df.groupby("image_id")["State"].first()
        merged = pivot.join(image_path_series, how="inner")
        merged = (
            merged.join(sampling_date_series, how="left")
            .join(height_series, how="left")
            .join(ndvi_series, how="left")
            .join(species_series, how="left")
            .join(state_series, how="left")
        )
        # Ensure all supervised primary targets are present
        merged = merged.dropna(subset=self.target_order)
        merged = merged.reset_index(drop=False)

        # Ensure all canonical biomass components are present as columns so that
        # downstream datasets can build ratio labels and 5D component targets.
        canonical_targets = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
        for t in canonical_targets:
            if t not in merged.columns:
                merged[t] = np.nan

        # Ensure proper column order: primary targets (target_order), then canonical targets,
        # followed by auxiliary labels and metadata.
        target_cols = []
        for t in list(self.target_order) + canonical_targets:
            if t not in target_cols and t in merged.columns:
                target_cols.append(t)
        cols = [
            *target_cols,
            "Height_Ave_cm",
            "Pre_GSHH_NDVI",
            "Species",
            "State",
            "Sampling_Date",
            "image_path",
            "image_id",
        ]
        merged = merged[cols]
        return merged

    def setup(self, stage: Optional[str] = None) -> None:
        # If predefined splits are provided, use them and skip random splitting
        if self._predefined_train_df is not None and self._predefined_val_df is not None:
            self.train_df = self._predefined_train_df.reset_index(drop=True)
            self.val_df = self._predefined_val_df.reset_index(drop=True)
            self._compute_and_store_zscore_stats()
            self._maybe_save_zscore_stats()
            return

        merged = self._read_and_pivot()
        rng = np.random.default_rng(seed=int(self.random_seed))
        indices = np.arange(len(merged))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * self.val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        self.train_df = merged.iloc[train_idx].reset_index(drop=True)
        self.val_df = merged.iloc[val_idx].reset_index(drop=True)
        self._compute_and_store_zscore_stats()
        self._maybe_save_zscore_stats()

    def build_full_dataframe(self) -> pd.DataFrame:
        return self._read_and_pivot()

    def train_dataloader(self) -> DataLoader:
        assert self.train_df is not None
        species_to_idx = self._ensure_species_mapping()
        state_to_idx = self._ensure_state_mapping()
        main_datasets: List[Dataset] = []

        csiro_ds = PastureImageDataset(
            records=self.train_df,
            root_dir=self.data_root,
            target_order=self.target_order,
            area_m2=self.sample_area_m2,
            reg3_mean=self._reg3_mean,
            reg3_std=self._reg3_std,
            ndvi_mean=self._ndvi_mean,
            ndvi_std=self._ndvi_std,
            log_scale_targets=self._log_scale_targets,
            species_to_idx=species_to_idx,
            state_to_idx=state_to_idx,
            transform=self.train_tf,
        )
        main_datasets.append(csiro_ds)

        # Optional Irish Glass Clover dataset mixed into the main regression stream
        if self.irish_enabled and self.irish_root and self.irish_csv and self.irish_train_tf is not None:
            irish_csv_path = os.path.join(self.irish_root, self.irish_csv)
            try:
                irish_ds = IrishGlassCloverDataset(
                    csv_path=irish_csv_path,
                    root_dir=self.irish_root,
                    image_subdir=self.irish_image_dir,
                    supervise_ratio=self.irish_supervise_ratio,
                    target_order=self.target_order,
                    reg3_mean=self._reg3_mean,
                    reg3_std=self._reg3_std,
                    log_scale_targets=self._log_scale_targets,
                    transform=self.irish_train_tf,
                )
                main_datasets.append(irish_ds)
            except Exception:
                # If anything goes wrong, fall back to CSIRO-only training
                pass

        if len(main_datasets) == 1:
            main_ds: Dataset = main_datasets[0]
        else:
            main_ds = ConcatDataset(main_datasets)

        # When manifold mixup is enabled and multiple datasets are concatenated, keep each batch
        # homogeneous (single dataset) so in-batch mixup never crosses datasets.
        if isinstance(main_ds, ConcatDataset) and bool(self._manifold_mixup_enabled):
            batch_sampler = _GroupedConcatBatchSampler(
                main_ds,
                batch_size=int(self.batch_size),
                shuffle=bool(self.shuffle),
            )
            main_loader = DataLoader(
                main_ds,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=bool(self.num_workers > 0),
            )
        else:
            main_loader = DataLoader(
                main_ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=bool(self.num_workers > 0),
            )
        if self.ndvi_dense_enabled and self._ndvi_cfg is not None:
            ndvi_loader = build_ndvi_scalar_dataloader(
                self._ndvi_cfg,
                split="train",
                transform=self.train_tf,
                reg3_dim=len(self.target_order),
                ndvi_mean=self._ndvi_mean,
                ndvi_std=self._ndvi_std,
            )
            return [main_loader, ndvi_loader]  # type: ignore[return-value]
        return main_loader

    def val_dataloader(self) -> DataLoader:
        assert self.val_df is not None
        species_to_idx = self._ensure_species_mapping()
        state_to_idx = self._ensure_state_mapping()
        ds = PastureImageDataset(
            records=self.val_df,
            root_dir=self.data_root,
            target_order=self.target_order,
            area_m2=self.sample_area_m2,
            reg3_mean=self._reg3_mean,
            reg3_std=self._reg3_std,
            ndvi_mean=self._ndvi_mean,
            ndvi_std=self._ndvi_std,
            log_scale_targets=self._log_scale_targets,
            species_to_idx=species_to_idx,
            state_to_idx=state_to_idx,
            transform=self.val_tf,
        )
        main_loader = DataLoader(
            ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=bool(self.num_workers > 0),
        )
        if self.ndvi_dense_enabled and self._ndvi_cfg is not None:
            ndvi_loader = build_ndvi_scalar_dataloader(
                self._ndvi_cfg,
                split="val",
                transform=self.val_tf,
                reg3_dim=len(self.target_order),
                ndvi_mean=self._ndvi_mean,
                ndvi_std=self._ndvi_std,
            )
            # Lightning supports multiple val loaders by returning a list
            return [main_loader, ndvi_loader]  # type: ignore[return-value]
        return main_loader

    

    # --- Auxiliary label utilities ---
    def _ensure_species_mapping(self) -> dict:
        if not hasattr(self, "_species_to_idx") or self._species_to_idx is None:
            df_all = self._read_and_pivot()
            uniques = sorted([str(s) for s in df_all["Species"].dropna().astype(str).unique().tolist()])
            mapping = {s: i for i, s in enumerate(uniques)}
            self._species_to_idx = mapping
        return self._species_to_idx

    @property
    def num_species_classes(self) -> int:
        mapping = self._ensure_species_mapping()
        return int(len(mapping))

    def _ensure_state_mapping(self) -> dict:
        if not hasattr(self, "_state_to_idx") or self._state_to_idx is None:
            df_all = self._read_and_pivot()
            uniques = sorted([str(s) for s in df_all["State"].dropna().astype(str).unique().tolist()])
            mapping = {s: i for i, s in enumerate(uniques)}
            self._state_to_idx = mapping
        return self._state_to_idx

    @property
    def num_state_classes(self) -> int:
        mapping = self._ensure_state_mapping()
        return int(len(mapping))

    # --- z-score helpers ---
    def _compute_and_store_zscore_stats(self) -> None:
        if self.train_df is None:
            return
        reg_means: List[float] = []
        reg_stds: List[float] = []
        area = float(self.sample_area_m2 if self.sample_area_m2 > 0.0 else 1.0)
        for t in self.target_order:
            vals = self.train_df[t].astype(float).to_numpy()
            vals = vals / area
            if self._log_scale_targets:
                vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
            mu = float(np.mean(vals)) if vals.size > 0 else 0.0
            sigma = float(np.std(vals)) if vals.size > 1 else 1.0
            if not np.isfinite(sigma) or sigma <= 0.0:
                sigma = 1.0
            reg_means.append(mu)
            reg_stds.append(sigma)
        self._reg3_mean = reg_means
        self._reg3_std = reg_stds

        # Compute z-score stats for canonical 5D biomass components in g/m^2:
        # [Dry_Clover_g, Dry_Dead_g, Dry_Green_g, GDM_g, Dry_Total_g]
        biomass_targets = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]
        b_means: List[float] = []
        b_stds: List[float] = []
        for t in biomass_targets:
            if t not in self.train_df.columns:
                # If this component is absent, fall back to standard normal
                b_means.append(0.0)
                b_stds.append(1.0)
                continue
            vals = self.train_df[t].astype(float).to_numpy()
            # Drop NaNs before statistics
            vals = vals[np.isfinite(vals)]
            vals = vals / area
            if self._log_scale_targets:
                vals = np.log1p(np.clip(vals, a_min=0.0, a_max=None))
            if vals.size == 0:
                mu, sigma = 0.0, 1.0
            else:
                mu = float(np.mean(vals))
                sigma = float(np.std(vals)) if vals.size > 1 else 1.0
                if not np.isfinite(sigma) or sigma <= 0.0:
                    sigma = 1.0
            b_means.append(mu)
            b_stds.append(sigma)
        self._biomass_5d_mean = b_means
        self._biomass_5d_std = b_stds
        try:
            ndvi_vals = self.train_df["Pre_GSHH_NDVI"].astype(float).to_numpy()
            ndvi_mu = float(np.mean(ndvi_vals)) if ndvi_vals.size > 0 else 0.0
            ndvi_sigma = float(np.std(ndvi_vals)) if ndvi_vals.size > 1 else 1.0
            if not np.isfinite(ndvi_sigma) or ndvi_sigma <= 0.0:
                ndvi_sigma = 1.0
        except Exception:
            ndvi_mu, ndvi_sigma = 0.0, 1.0
        self._ndvi_mean = ndvi_mu
        self._ndvi_std = ndvi_sigma

    def _maybe_save_zscore_stats(self) -> None:
        if not self._zscore_output_path:
            return
        try:
            out_dir = os.path.dirname(self._zscore_output_path)
            if out_dir and not os.path.isdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            import json
            payload = {
                "reg3": {
                    "mean": list(self._reg3_mean or []),
                    "std": list(self._reg3_std or []),
                },
                "ndvi": {
                    "mean": float(self._ndvi_mean if self._ndvi_mean is not None else 0.0),
                    "std": float(self._ndvi_std if self._ndvi_std is not None else 1.0),
                },
                # Optional 5D biomass stats (g/m^2, possibly log-transformed)
                "biomass_5d": {
                    "mean": list(self._biomass_5d_mean or []),
                    "std": list(self._biomass_5d_std or []),
                    "order": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"],
                },
                "meta": {
                    "area_m2": float(self.sample_area_m2),
                    "log_scale_targets": bool(self._log_scale_targets),
                    "target_order": list(self.target_order),
                },
            }
            with open(self._zscore_output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    @property
    def reg3_zscore_mean(self) -> Optional[List[float]]:
        return self._reg3_mean

    @property
    def reg3_zscore_std(self) -> Optional[List[float]]:
        return self._reg3_std

    @property
    def ndvi_zscore_mean(self) -> Optional[float]:
        return self._ndvi_mean

    @property
    def ndvi_zscore_std(self) -> Optional[float]:
        return self._ndvi_std

    @property
    def biomass_5d_zscore_mean(self) -> Optional[List[float]]:
        return self._biomass_5d_mean

    @property
    def biomass_5d_zscore_std(self) -> Optional[List[float]]:
        return self._biomass_5d_std
