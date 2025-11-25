"""
简单的 probing 提交脚本：对每个 target_name 使用 CSIRO 训练集的全局均值作为预测。

提交行为与 `infer_and_submit_pt.py` 保持一致：
- 通过 `INPUT_PATH` 查找 `test.csv`（可以是目录或直接是 test.csv 路径）
- 输出到 `OUTPUT_SUBMISSION_PATH`（默认与原脚本一致为 "submission.csv"）
- 提交文件格式为两列：sample_id,target
"""

# ===== Required user variables (与 infer_and_submit_pt.py 对齐) =====
# INPUT_PATH 可以是：
#   1) 含有 test.csv 的目录（例如 "data"）
#   2) 直接指向 test.csv 的路径（例如 "data/test.csv"）
INPUT_PATH = "data"
OUTPUT_SUBMISSION_PATH = "submission.csv"
# ==========================================================

import os
import pandas as pd


# 由 compute_train_means.py 基于 data/train.csv 计算得到的均值（CSIRO 训练集）
MEAN_BY_TARGET = {
    "Dry_Clover_g": 6.649692156862745,
    "Dry_Dead_g": 12.04454761904762,
    "Dry_Green_g": 26.62472240896359,
    "Dry_Total_g": 45.31809663865546,
    "GDM_g": 33.2744137254902,
}


def resolve_paths(input_path: str):
    """
    与 infer_and_submit_pt.py 中的 resolve_paths 行为保持一致。
    """
    if os.path.isdir(input_path):
        dataset_root = input_path
        test_csv = os.path.join(input_path, "test.csv")
    else:
        dataset_root = os.path.dirname(os.path.abspath(input_path))
        test_csv = input_path
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    return dataset_root, test_csv


def main() -> None:
    # 与 infer_and_submit_pt.py 一样，通过 INPUT_PATH 解析 test.csv
    _, test_csv = resolve_paths(INPUT_PATH)

    df_test = pd.read_csv(test_csv)
    required_cols = {"sample_id", "image_path", "target_name"}
    missing = required_cols - set(df_test.columns)
    if missing:
        raise ValueError(f"test.csv must contain columns: {missing}")

    unknown_targets = set(df_test["target_name"].unique()) - set(MEAN_BY_TARGET.keys())
    if unknown_targets:
        raise ValueError(f"Found unexpected target_name(s) in test.csv: {unknown_targets}")

    # 构造提交：每一行根据 target_name 填充均值
    rows = []
    for _, r in df_test.iterrows():
        sample_id = str(r["sample_id"])
        target_name = str(r["target_name"])
        value = MEAN_BY_TARGET.get(target_name)
        if value is None:
            raise KeyError(f"Unknown target_name encountered: {target_name}")
        rows.append((sample_id, float(value)))

    out_path = OUTPUT_SUBMISSION_PATH
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("sample_id,target\n")
        for sample_id, value in rows:
            f.write(f"{sample_id},{value}\n")

    print(f"Submission written to: {OUTPUT_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

