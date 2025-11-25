import argparse
import os
import pandas as pd

# CSIRO 训练集均值 (from compute_train_means.py)
TRAIN_MEANS = {
    "Dry_Clover_g": 6.649692156862745,
    "Dry_Dead_g": 12.04454761904762,
    "Dry_Green_g": 26.62472240896359,
    "Dry_Total_g": 45.31809663865546,
    "GDM_g": 33.2744137254902,
}

def resolve_paths(input_path: str):
    if os.path.isdir(input_path):
        test_csv = os.path.join(input_path, "test.csv")
    else:
        test_csv = input_path
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    return test_csv

def main():
    parser = argparse.ArgumentParser(description="生成缩放均值的 Probing 提交文件")
    parser.add_argument("--scale", type=float, required=True, help="均值缩放系数 (例如 0.6 或 1.4)")
    parser.add_argument("--target", type=str, default="all", 
                        help="指定缩放的 target_name，默认 'all' 表示所有列都缩放。如果指定某个列，其他列保持 1.0 倍均值")
    parser.add_argument("--input_path", default="data", help="输入路径，可以是包含 test.csv 的目录或文件路径")
    parser.add_argument("--out", required=True, help="输出的 submission csv 路径")
    
    args = parser.parse_args()

    test_csv_path = resolve_paths(args.input_path)
    df = pd.read_csv(test_csv_path)
    
    # 准备预测值
    rows = []
    print(f"Generating probing submission with Scale={args.scale} for Target={args.target}...")
    
    for _, r in df.iterrows():
        sample_id = str(r["sample_id"])
        t_name = str(r["target_name"])
        
        base_val = TRAIN_MEANS.get(t_name, 0.0)
        
        # 决定是否应用缩放
        if args.target == "all" or args.target == t_name:
            final_val = base_val * args.scale
        else:
            final_val = base_val # 保持默认均值
            
        rows.append((sample_id, final_val))

    # 写入文件
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("sample_id,target\n")
        for sid, val in rows:
            f.write(f"{sid},{val}\n")
            
    print(f"Done. Saved to {args.out}")

if __name__ == "__main__":
    main()

