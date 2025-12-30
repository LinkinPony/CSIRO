#!/bin/bash
# 1. 获取脚本所在的绝对路径
# 这样无论你在哪个目录下执行 source，它都能找到同目录下的 token.secret
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SECRET_FILE="$SCRIPT_DIR/token.secret"

# 2. 检查文件是否存在
if [ -f "$SECRET_FILE" ]; then
    # 3. 读取内容并赋值给变量
    # 使用 cat 读取，并利用 bash 特性自动去除末尾换行符
    # 如果文件中有特殊字符，建议加上引号
    TOKEN_CONTENT=$(<"$SECRET_FILE")
    
    # 4. 导出环境变量
    export KAGGLE_API_TOKEN="$TOKEN_CONTENT"
    
    echo "✅ KAGGLE_API_TOKEN 已成功设置 (来源: $SECRET_FILE)"
else
    echo "❌ 错误: 在 $SCRIPT_DIR 下未找到 token.secret 文件"
    return 1 2>/dev/null || exit 1
fi
# --- 配置 ---

# 1. 设置包含数据集文件的文件夹路径
#    (请确保 "weights" 是相对于您运行此脚本的正确路径)
DATASET_PATH="weights"

# 2. 设置您的基础版本信息
BASE_MESSAGE=" "

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
  VERSION_MESSAGE="$CONFIG_VERSION at $CURRENT_TIME"
else
  VERSION_MESSAGE="$BASE_MESSAGE at $CURRENT_TIME"
fi

MAX=100
VERSION_MESSAGE="$(echo -n "$VERSION_MESSAGE" | cut -c1-$MAX)"
echo "Message(len=$(echo -n "$VERSION_MESSAGE" | wc -c)): $VERSION_MESSAGE"
# 6. 执行 Kaggle CLI 命令
echo "    Path: $DATASET_PATH"
echo "    Config version: ${CONFIG_VERSION:-N/A}"
echo "    Message: $VERSION_MESSAGE"
echo "------------------------------------------------"

kaggle datasets version -p "$DATASET_PATH" -m "$VERSION_MESSAGE" --dir-mode zip


echo "------------------------------------------------"