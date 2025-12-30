from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd


def read_and_pivot_csiro_train_csv(
    *,
    data_root: str,
    train_csv: str,
    target_order: Sequence[str],
) -> pd.DataFrame:
    """
    Read CSIRO `train.csv` and pivot it into an image-level dataframe.

    IMPORTANT: This function is intentionally kept **bit-for-bit aligned** with
    the project's historical implementation (`PastureDataModule._read_and_pivot`)
    so that all experiments share the exact same aggregation semantics.
    """

    csv_path = os.path.join(str(data_root), str(train_csv))
    df = pd.read_csv(csv_path)

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
    merged = merged.dropna(subset=list(target_order))
    merged = merged.reset_index(drop=False)

    # Ensure all canonical biomass components are present as columns
    canonical_targets = [
        "Dry_Green_g",
        "Dry_Dead_g",
        "Dry_Clover_g",
        "GDM_g",
        "Dry_Total_g",
    ]
    for t in canonical_targets:
        if t not in merged.columns:
            merged[t] = np.nan

    # Ensure proper column order: primary targets (target_order), then canonical targets,
    # followed by auxiliary labels and metadata.
    target_cols: list[str] = []
    for t in list(target_order) + canonical_targets:
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


