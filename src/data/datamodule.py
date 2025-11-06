import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule


@dataclass
class NormalizationSpec:
    mean: Sequence[float]
    std: Sequence[float]


def build_transforms(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
    train_scale: Tuple[float, float] = (0.8, 1.0),
    hflip_prob: float = 0.5,
):
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=train_scale),
            T.RandomHorizontalFlip(p=hflip_prob),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, val_tf


class PastureImageDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        root_dir: str,
        target_order: Sequence[str],
        species_to_idx: Optional[dict] = None,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.records = records.reset_index(drop=True)
        self.root_dir = root_dir
        self.target_order = list(target_order)
        self.species_to_idx = dict(species_to_idx or {})
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records.iloc[index]
        image_path = os.path.join(self.root_dir, str(row["image_path"]))
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        # main regression targets (triple)
        y_reg3 = torch.tensor([row[t] for t in self.target_order], dtype=torch.float32)
        # auxiliary regression target: height
        y_height = torch.tensor([float(row.get("Height_Ave_cm", 0.0))], dtype=torch.float32)
        # auxiliary regression target: Pre_GSHH_NDVI
        y_ndvi = torch.tensor([float(row.get("Pre_GSHH_NDVI", 0.0))], dtype=torch.float32)
        # auxiliary classification target: species index
        species = str(row.get("Species", ""))
        if self.species_to_idx and species in self.species_to_idx:
            y_species = torch.tensor(int(self.species_to_idx[species]), dtype=torch.long)
        else:
            y_species = torch.tensor(0, dtype=torch.long)
        return {"image": image, "y_reg3": y_reg3, "y_height": y_height, "y_ndvi": y_ndvi, "y_species": y_species}


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
        hflip_prob: float = 0.5,
        shuffle: bool = True,
        prefetch_factor: int = 2,
        predefined_train_df: Optional[pd.DataFrame] = None,
        predefined_val_df: Optional[pd.DataFrame] = None,
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
        self.train_tf, self.val_tf = build_transforms(
            image_size=image_size, mean=mean, std=std, train_scale=train_scale, hflip_prob=hflip_prob
        )

        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self._predefined_train_df: Optional[pd.DataFrame] = predefined_train_df
        self._predefined_val_df: Optional[pd.DataFrame] = predefined_val_df

    def _read_and_pivot(self) -> pd.DataFrame:
        csv_path = os.path.join(self.data_root, self.train_csv)
        df = pd.read_csv(csv_path)
        df = df[df["target_name"].isin(self.target_order)]
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
        height_series = df.groupby("image_id")["Height_Ave_cm"].first()
        ndvi_series = df.groupby("image_id")["Pre_GSHH_NDVI"].first()
        species_series = df.groupby("image_id")["Species"].first()
        merged = pivot.join(image_path_series, how="inner")
        merged = merged.join(height_series, how="left").join(ndvi_series, how="left").join(species_series, how="left")
        merged = merged.dropna(subset=self.target_order)
        merged = merged.reset_index(drop=False)
        # Ensure proper column order: targets then image_path and image_id
        cols = [*self.target_order, "Height_Ave_cm", "Pre_GSHH_NDVI", "Species", "image_path", "image_id"]
        merged = merged[cols]
        return merged

    def setup(self, stage: Optional[str] = None) -> None:
        # If predefined splits are provided, use them and skip random splitting
        if self._predefined_train_df is not None and self._predefined_val_df is not None:
            self.train_df = self._predefined_train_df.reset_index(drop=True)
            self.val_df = self._predefined_val_df.reset_index(drop=True)
            return

        merged = self._read_and_pivot()
        rng = np.random.default_rng(seed=42)
        indices = np.arange(len(merged))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * self.val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        self.train_df = merged.iloc[train_idx].reset_index(drop=True)
        self.val_df = merged.iloc[val_idx].reset_index(drop=True)

    def build_full_dataframe(self) -> pd.DataFrame:
        return self._read_and_pivot()

    def train_dataloader(self) -> DataLoader:
        assert self.train_df is not None
        species_to_idx = self._ensure_species_mapping()
        ds = PastureImageDataset(
            records=self.train_df,
            root_dir=self.data_root,
            target_order=self.target_order,
            species_to_idx=species_to_idx,
            transform=self.train_tf,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=bool(self.num_workers > 0),
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_df is not None
        species_to_idx = self._ensure_species_mapping()
        ds = PastureImageDataset(
            records=self.val_df,
            root_dir=self.data_root,
            target_order=self.target_order,
            species_to_idx=species_to_idx,
            transform=self.val_tf,
        )
        return DataLoader(
            ds,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=bool(self.num_workers > 0),
        )

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


