#!/bin/bash

# ================================================================
# Hybrid Pipeline: 
# 1. 视频 -> 抽帧 (间隔 N) -> RAFT 分离前景/背景
# 2. 背景: 只取指定的一帧 (默认第0帧) 生成静态 MPI Atlas
# 3. 前景: 所有提取出的前景帧生成动态 MPI Atlas 序列
# 4. 自动部署并生成预览链接
#
# 用法:
#   ./scripts/run_hybrid_pipeline.sh <video_path> <scene_name> [bg_frame_index]
# ================================================================

set -eo pipefail

# 路径配置
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 参数解析
VIDEO_PATH=${1:-}
BASE_SCENE_NAME=${2:-hybrid_scene}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCENE_NAME="${BASE_SCENE_NAME}_${TIMESTAMP}"
BG_FRAME_INDEX=${3:-0}  # 默认使用第0帧作为静态背景
MAX_FRAMES=${4:--1}     # 默认处理所有帧
FRAME_INTERVAL=${5:-5}  # 默认抽帧间隔为5

if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: $0 <video_path> <scene_name> [bg_frame_index] [max_frames] [frame_interval]"
    echo "Example: $0 assets/campus360.mp4 campus_hybrid 0 100 5"
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
echo "[Step 1/4] RAFT 提取前景与背景 (间隔: ${FRAME_INTERVAL})"
echo "==============================================================="
# 激活 nerfstudio 环境用于 RAFT
source "$CONDA_ACTIVATE"/bin/activate nerfstudio

# 注意: extract 命令会生成 background_frames, masks, foreground_rgba
python "$PROJECT_ROOT/main.py" extract \
    --video "$VIDEO_PATH" \
    --output "$WORK_DIR" \
    --width $WIDTH --height $HEIGHT \
    --mask_method raft \
    --raft_model "$RAFT_MODEL" \
    --frame_interval $FRAME_INTERVAL \
    --save_foreground \
    --mask_threshold 0.2 \
    --mask_kernel 3 \
    --inpaint_radius 3 \
    --max_frames "$MAX_FRAMES"

echo "==============================================================="
echo "[Step 2/4] 生成静态背景 MPI (使用第 ${BG_FRAME_INDEX} 帧)"
echo "==============================================================="
# 激活 panosynthvr-py39 环境用于 MPI 生成
source "$CONDA_ACTIVATE"/bin/activate panosynthvr-py39

# 我们只需要处理指定的那一帧背景
# 为了节省时间，我们创建一个临时的文件夹，只包含那一帧
TEMP_BG_DIR="$WORK_DIR/temp_static_bg"
mkdir -p "$TEMP_BG_DIR"

# 找到对应的帧文件 (格式 frame_XXXXXX.png)
TARGET_FRAME_NAME=$(printf "frame_%06d.png" $BG_FRAME_INDEX)
SRC_BG_PATH="$WORK_DIR/background_frames/$TARGET_FRAME_NAME"

if [[ ! -f "$SRC_BG_PATH" ]]; then
    echo "Error: 指定的背景帧 $SRC_BG_PATH 不存在！"
    echo "请检查 bg_frame_index 是否在提取的帧范围内 (注意 frame_interval)"
    exit 1
fi

cp "$SRC_BG_PATH" "$TEMP_BG_DIR/"

# 生成背景 Atlas
python "$PROJECT_ROOT/main.py" background \
    --frames_dir "$TEMP_BG_DIR" \
    --output "$WORK_DIR/background_atlas" \
    --width $WIDTH --height $HEIGHT \
    --manifest "$WORK_DIR/background_manifest.json"

echo "==============================================================="
echo "[Step 3/4] 生成动态前景 MPI (所有帧)"
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
echo "[Step 4/4] 部署与索引更新"
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

echo "==============================================================="
echo "Hybrid Pipeline 完成!"
echo "==============================================================="
echo "场景名称: $SCENE_NAME"
echo "背景帧: #$BG_FRAME_INDEX (静态)"
echo "前景: 动态序列 (间隔 $FRAME_INTERVAL)"
echo ""
echo "访问地址: http://127.0.0.1:3600/docs/index.html"
echo "注意: 启动时请确保 URL 参数包含 &bg_static=true (Viewer 默认支持，但最好确认一下)"
