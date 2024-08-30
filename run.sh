#!/bin/bash
model_version=$1
frame_number=$2

# Print the arguments (optional, for debugging)
echo "Model version: $model_version"
echo "Frame number: $frame_number"


# 定义项目的根目录变量
PROJECT_ROOT_DIR="$(dirname "$(readlink -f "$0")")"

# 创建新的 build 目录并进入
cd "$PROJECT_ROOT_DIR/build" && ./FunctionTest $model_version $frame_number

# 检查上一个命令是否成功
if [ $? -eq 0 ]; then
    echo "Run succeeded."
else
    echo "Run failed."
    exit 1
fi