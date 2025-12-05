#!/bin/bash

# ================================================================
# Hybrid Pipeline (Background Subtraction): 
# 1. 视频 -> 抽帧 (间隔 N) -> Background Subtraction 分离前景/背景
# 2. 背景: 使用 Temporal Median 生成的静态背景
# 3. 前景: 差分法提取出的前景生成动态 MPI Atlas 序列
# 4. 自动部署并生成预览链接
#
# 用法:
#   ./scripts/run_hybrid_bgsub.sh <video_path> <scene_name> [bg_frame_index] [max_frames] [frame_interval]
# ================================================================

set -eo pipefail

# 路径配置
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 记录开始时间
START_TIME=$(date +%s)

# 参数解析
VIDEO_PATH=${1:-}
BASE_SCENE_NAME=${2:-hybrid_bgsub_scene}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCENE_NAME="${BASE_SCENE_NAME}_${TIMESTAMP}"
BG_FRAME_INDEX=${3:-0}  # 默认使用第0帧作为静态背景
MAX_FRAMES=${4:--1}     # 默认处理所有帧
FRAME_INTERVAL=${5:-5}  # 默认抽帧间隔为5

if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: $0 <video_path> <scene_name> [bg_frame_index] [max_frames] [frame_interval]"
    echo "Example: $0 assets/campus360.mp4 campus_bgsub 0 100 5"
    echo "Note: Output will be saved to .../${SCENE_NAME}"
    exit 1
fi

# ----------- 配置参数 -----------
WIDTH=1024
HEIGHT=512
# FRAME_INTERVAL is set via argument above
RAFT_MODEL="$PROJECT_ROOT/submodules/RAFT/models/raft-small.pth"
WORK_DIR="$PROJECT_ROOT/output/hybrid_output/${SCENE_NAME}"
ASSET_DIR="$PROJECT_ROOT/docs/assets/${SCENE_NAME}"
CONDA_ACTIVATE="/home/ksi/miniconda3"

# 清理旧数据
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "==============================================================="
echo "[Step 1/5] 生成静态背景 (Temporal Median)"
echo "==============================================================="
# 激活 panosynthvr-py39 环境
if [ -f "$CONDA_ACTIVATE/bin/activate" ]; then
    source "$CONDA_ACTIVATE"/bin/activate panosynthvr-py39
fi

# 创建临时文件夹存放这一张干净的背景图
TEMP_BG_DIR="$WORK_DIR/temp_static_bg"
mkdir -p "$TEMP_BG_DIR"

CLEAN_BG_PATH="$TEMP_BG_DIR/frame_000000.png"

# 1. 生成干净背景
python "$PROJECT_ROOT/main.py" clean_bg \
    --video "$VIDEO_PATH" \
    --output "$CLEAN_BG_PATH" \
    --width $WIDTH --height $HEIGHT \
    --sample_count 50

if [[ ! -f "$CLEAN_BG_PATH" ]]; then
    echo "Error: 无法生成干净背景 $CLEAN_BG_PATH"
    exit 1
fi

echo "==============================================================="
echo "[Step 2/5] 提取前景 (Background Subtraction)"
echo "==============================================================="
# 激活 panosynthvr-py39 环境
if [ -f "$CONDA_ACTIVATE/bin/activate" ]; then
    source "$CONDA_ACTIVATE"/bin/activate nerfstudio
fi

# 注意: extract 命令会生成 background_frames, masks, foreground_rgba
# 使用 background_subtraction 方法，阈值设为 25 (对于 0-255 的像素差)
python "$PROJECT_ROOT/main.py" extract \
    --video "$VIDEO_PATH" \
    --output "$WORK_DIR" \
    --width $WIDTH --height $HEIGHT \
    --mask_method background_subtraction \
    --bg_image "$CLEAN_BG_PATH" \
    --frame_interval $FRAME_INTERVAL \
    --save_foreground \
    --mask_threshold 25 \
    --mask_kernel 3 \
    --inpaint_radius 3 \
    --max_frames "$MAX_FRAMES"

echo "==============================================================="
echo "[Step 3/5] 生成静态背景 MPI"
echo "==============================================================="
if [ -f "$CONDA_ACTIVATE/bin/activate" ]; then
    source "$CONDA_ACTIVATE"/bin/activate panosynthvr-py39
fi

# 使用刚才生成的干净背景作为 MPI 的源
python "$PROJECT_ROOT/main.py" background \
    --frames_dir "$TEMP_BG_DIR" \
    --output "$WORK_DIR/background_atlas" \
    --width $WIDTH --height $HEIGHT \
    --manifest "$WORK_DIR/background_manifest.json"

echo "==============================================================="
echo "[Step 4/5] 生成动态前景 MPI"
echo "==============================================================="
# 继续使用 panosynthvr-py39 环境
python "$PROJECT_ROOT/main.py" foreground \
    --input_dir "$WORK_DIR/foreground_rgba" \
    --output "$WORK_DIR/foreground_atlas" \
    --width $WIDTH --height $HEIGHT \
    --manifest "$WORK_DIR/foreground_manifest.json"

# 修正前景 Manifest 的 basePath
# 因为前景 Atlas 被移动到了 foreground_atlas 子目录，所以需要更新 manifest
if [[ -f "$WORK_DIR/foreground_manifest.json" ]]; then
    sed -i 's/"basePath": ""/"basePath": "foreground_atlas\/"/' "$WORK_DIR/foreground_manifest.json"
fi

echo "==============================================================="
echo "[Step 5/5] 部署与索引更新"
echo "==============================================================="
rm -rf "$ASSET_DIR"
mkdir -p "$ASSET_DIR"

# 拷贝背景 (因为只有一帧，所以只有一个文件夹)
cp -r "$WORK_DIR/background_atlas"/frame_* "$ASSET_DIR"/
cp "$WORK_DIR/background_manifest.json" "$ASSET_DIR/frames_manifest.json"

# 拷贝前景 (所有帧)
mkdir -p "$ASSET_DIR/foreground_atlas"
cp -r "$WORK_DIR/foreground_atlas"/frame_* "$ASSET_DIR/foreground_atlas"/
cp "$WORK_DIR/foreground_manifest.json" "$ASSET_DIR/foreground_manifest.json"

# 更新索引页
python "$PROJECT_ROOT/main.py" index

# 记录结束时间并计算耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

TIMESTAMP=$(date +%s)

# 计算实际生成的帧数
ACTUAL_FRAMES=$(ls "$ASSET_DIR/foreground_atlas" | wc -l)

echo "==============================================================="
echo "Hybrid Pipeline (Background Subtraction) 完成!"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "==============================================================="
echo "场景名称: $SCENE_NAME"
echo "背景帧: #0 (静态)"
echo "前景: 动态序列 (间隔 $FRAME_INTERVAL)"
echo "实际生成帧数: $ACTUAL_FRAMES"
echo ""
echo "访问地址: http://127.0.0.1:3600/docs/renderer.html?mode=hybrid&name=$SCENE_NAME&frames=$ACTUAL_FRAMES&fps=24&t=$TIMESTAMP"
echo "注意: URL 已自动更新为实际帧数 ($ACTUAL_FRAMES)"
