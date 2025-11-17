#!/usr/bin/env python3
"""
Compute the covariance matrix for selected target variables from data/train.csv.

The input CSV is in long format with columns including:
  - image_path
  - target_name
  - target

This script pivots the data to wide format using `image_path` as the index,
with each requested `target_name` becoming a column (values from `target`),
then computes and prints the covariance matrix of those columns.
"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_COLUMNS: List[str] = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute covariance matrix for selected targets by pivoting train.csv."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/train.csv",
        help="Path to the input CSV (default: data/train.csv)",
    )
    parser.add_argument(
        "--index-key",
        type=str,
        default="image_path",
        help="Column used as pivot index key (default: image_path)",
    )
    parser.add_argument(
        "--cols",
        nargs="*",
        default=DEFAULT_COLUMNS,
        help="Target columns (target_name values) to include (default: 5 standard targets).",
    )
    parser.add_argument(
        "--keep-na",
        dest="dropna",
        action="store_false",
        help="Keep rows with any NaNs in selected columns (default: drop).",
    )
    parser.add_argument(
        "--dropna",
        dest="dropna",
        action="store_true",
        help="Drop rows with any NaNs in selected columns (default).",
    )
    parser.set_defaults(dropna=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the covariance matrix as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"target_name", "target", args.index_key}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(
            f"Input CSV is missing required columns: {missing_str}. "
            f"Present columns: {', '.join(df.columns)}"
        )

    # Filter to only the requested target_name categories
    selected = df[df["target_name"].isin(args.cols)].copy()
    if selected.empty:
        raise ValueError(
            "No rows found for requested target_name values: "
            f"{', '.join(args.cols)}"
        )

    # Pivot to wide format: one row per index_key, columns by target_name, values from target
    wide = (
        selected.pivot_table(
            index=args.index_key,
            columns="target_name",
            values="target",
            aggfunc="mean",
        )
        .reset_index()
        .set_index(args.index_key)
    )

    # Ensure all requested columns are present in the pivot result
    missing_cols = [c for c in args.cols if c not in wide.columns]
    if missing_cols:
        raise ValueError(
            "The pivoted table is missing columns: "
            f"{', '.join(missing_cols)}. Available: {', '.join(map(str, wide.columns))}"
        )

    # Reorder and optionally drop rows with any NaN
    wide = wide[args.cols]
    if args.dropna:
        wide = wide.dropna(how="any")

    if wide.shape[0] < 2:
        raise ValueError(
            f"Not enough rows after pivot/dropna to compute covariance (rows={wide.shape[0]})."
        )

    cov = wide.cov(ddof=1)

    # Print nicely to stdout
    print(f"Selected columns: {', '.join(args.cols)}")
    print(f"Num samples used: {wide.shape[0]}")
    print("Covariance matrix:")
    print(cov.to_string(float_format=lambda x: f"{x:.6f}"))

    # Optional save
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cov.to_csv(out_path, float_format="%.6f")
        print(f"\nSaved covariance matrix to: {out_path.resolve()}")


if __name__ == "__main__":
    main()


