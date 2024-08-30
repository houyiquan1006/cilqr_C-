#!/bin/bash
model_version=$1
frame_number=$2
# 定义项目的根目录变量
PROJECT_ROOT_DIR="$(dirname "$(readlink -f "$0")")"

# 删除旧的 build 目录
rm -rf "$PROJECT_ROOT_DIR/build/"

# 创建新的 build 目录并进入
mkdir "$PROJECT_ROOT_DIR/build" && cd "$PROJECT_ROOT_DIR/build"

# 运行 CMake 和 Make 来编译项目
cmake .. && make

# 检查上一个命令是否成功
if [ $? -eq 0 ]; then
    echo "Build succeeded."
    output_path="./tmp/${model_version}_${frame_number}/log.txt"
    cd .. && rm -rf ./log.txt && bash run.sh $model_version $frame_number > "$output_path"
else
    echo "Build failed."
    exit 1
fi