import argparse
import os
import re

import matplotlib.pyplot as plt
import pandas as pd


def sanitize_filename(name: str) -> str:
    """Convert a column name into a safe filename fragment."""
    # Replace anything that is not alphanumeric, dash, underscore or dot
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def plot_distributions(
    csv_path: str,
    output_dir: str,
    bins: int = 30,
    max_categories: int = 50,
) -> None:
    """Plot distribution for every column in the CSV.

    - Numeric columns: histogram
    - Non-numeric columns: bar plot of value counts (top N categories)
    - Additionally, if the CSV is in the 'flattened' format with
      columns `target_name` and `target`, extra plots are created
      for each target (grouped by `target_name`).
    """
    df = pd.read_csv(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    for idx, col in enumerate(df.columns):
        series = df[col]

        # Skip columns that are completely empty
        if series.dropna().empty:
            continue

        plt.figure(figsize=(7, 5))

        if pd.api.types.is_numeric_dtype(series):
            # Numeric: histogram
            series.plot.hist(bins=bins, edgecolor="black")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {col}")
        else:
            # Categorical / string-like: bar plot of counts
            value_counts = series.astype(str).value_counts()
            if len(value_counts) > max_categories:
                value_counts = value_counts.head(max_categories)
                title_suffix = f" (top {max_categories})"
            else:
                title_suffix = ""

            value_counts.plot.bar()
            plt.ylabel("Count")
            plt.title(f"Category counts of {col}{title_suffix}")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        safe_col = sanitize_filename(col)
        filename = f"{idx:02d}_{safe_col}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        plt.close()

    # Special handling for flattened targets: plot `target` per `target_name`
    if {"target", "target_name"}.issubset(df.columns):
        target_col = "target"
        group_col = "target_name"

        for target_name, group_df in df.groupby(group_col):
            series = group_df[target_col]
            if series.dropna().empty:
                continue

            plt.figure(figsize=(7, 5))
            series.plot.hist(bins=bins, edgecolor="black")
            plt.xlabel(f"{target_col} ({target_name})")
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {target_col} for {target_name}")
            plt.tight_layout()

            safe_tn = sanitize_filename(str(target_name))
            filename = f"target_{safe_tn}.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot distributions for all columns in a CSV file.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/train.csv",
        help="Path to the input CSV file (default: data/train.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train_distributions",
        help="Directory to save distribution plots (default: outputs/train_distributions).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Number of bins for numeric histograms (default: 30).",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=50,
        help="Maximum number of categories to display for non-numeric columns (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_distributions(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        bins=args.bins,
        max_categories=args.max_categories,
    )


if __name__ == "__main__":
    main()


