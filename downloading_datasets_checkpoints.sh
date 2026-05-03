#!/bin/bash

# 检查并创建 checkpoints 目录
mkdir -p ./checkpoints

# 下载 FocusUI-3B 模型
echo "正在下载 FocusUI-3B..."
hf download yyyang/FocusUI-3B --repo-type model --local-dir ./checkpoints/FocusUI-3B

# 下载 FocusUI-7B 模型
echo "正在下载 FocusUI-7B..."
hf download yyyang/FocusUI-7B --repo-type model --local-dir ./checkpoints/FocusUI-7B

# 检查并创建 datasets 目录
mkdir -p ./datasets

# 下载 UI-Grounding-Benchmarks 数据集
echo "正在下载 UI-Grounding-Benchmarks..."
hf download yyyang/UI-Grounding-Benchmarks --repo-type dataset --local-dir ./datasets/UI-Grounding-Benchmarks

echo "所有文件下载完成！"