#!/bin/bash

# --- 配置 ---

# 1. 设置包含数据集文件的文件夹路径
#    (请确保 "weights" 是相对于您运行此脚本的正确路径)
DATASET_PATH="weights"

# 2. 设置您的基础版本信息
BASE_MESSAGE="update: new checkpoints"

# --- 脚本执行 ---

# 3. 获取当前时间戳 (格式: 年-月-日 时:分:秒)
CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 4. 组合最终的版本信息
#    注意：Kaggle 的 -m 消息有100个字符的限制，请保持信息简洁
VERSION_MESSAGE="$BASE_MESSAGE at $CURRENT_TIME"

# 5. 执行 Kaggle CLI 命令
echo "    Path: $DATASET_PATH"
echo "    Message: $VERSION_MESSAGE"
echo "------------------------------------------------"

kaggle datasets version -p "$DATASET_PATH" -m "$VERSION_MESSAGE" --dir-mode zip


echo "------------------------------------------------"