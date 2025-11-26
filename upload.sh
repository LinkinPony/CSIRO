#!/bin/bash

# --- 配置 ---

# 1. 设置包含数据集文件的文件夹路径
#    (请确保 "weights" 是相对于您运行此脚本的正确路径)
DATASET_PATH="weights"

# 2. 设置您的基础版本信息
BASE_MESSAGE="update: new checkpoints"

# 3. 从配置文件中读取 version 字段 (weights/configs/train.yaml)
CONFIG_FILE="$DATASET_PATH/configs/train.yaml"
CONFIG_VERSION=""
if [ -f "$CONFIG_FILE" ]; then
  CONFIG_VERSION=$(grep -E '^[[:space:]]*version:' "$CONFIG_FILE" | head -n1 | cut -d':' -f2- | xargs)
fi

# --- 脚本执行 ---

# 4. 获取当前时间戳 (格式: 年-月-日 时:分:秒)
CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")

# 5. 组合最终的版本信息
#    注意：Kaggle 的 -m 消息有100个字符的限制，请保持信息简洁
if [ -n "$CONFIG_VERSION" ]; then
  VERSION_MESSAGE="$BASE_MESSAGE (version: $CONFIG_VERSION) at $CURRENT_TIME"
else
  VERSION_MESSAGE="$BASE_MESSAGE at $CURRENT_TIME"
fi

# 6. 执行 Kaggle CLI 命令
echo "    Path: $DATASET_PATH"
echo "    Config version: ${CONFIG_VERSION:-N/A}"
echo "    Message: $VERSION_MESSAGE"
echo "------------------------------------------------"

kaggle datasets version -p "$DATASET_PATH" -m "$VERSION_MESSAGE" --dir-mode zip


echo "------------------------------------------------"