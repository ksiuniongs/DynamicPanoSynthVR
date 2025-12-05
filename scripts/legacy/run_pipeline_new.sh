#!/bin/bash

# ================================================================
# End-to-end pipeline:
# 1. 前景/背景分离（RAFT）
# 2. 生成背景 MPI atlas + manifest
# 3. 生成前景 MPI atlas + manifest
# 4. 拷贝到 docs/assets/<scene>，直接用 atlas_sequence_viewer.html 预览
#
# 用法:
#   ./run_pipeline.sh <video_path> <scene_name> [max_frames]
# 示例:
#   ./run_pipeline.sh campus360.mp4 campus_run_1024 120
# ================================================================

set -euo pipefail
conda_path="/home/ksi/miniconda3"
VIDEO_PATH=${1:-}
SCENE_NAME=${2:-pipeline_scene}
MAX_FRAMES=${3:--1}

if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: $0 <video_path> <scene_name> [max_frames]"
    exit 1
fi

# ----------- 配置 -----------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

WIDTH=2048
HEIGHT=1024
FRAME_INTERVAL=5
WORK_DIR="$PROJECT_ROOT/output/pipeline_output/${SCENE_NAME}"
ASSET_DIR="$PROJECT_ROOT/docs/assets/${SCENE_NAME}"
RAFT_MODEL="$PROJECT_ROOT/submodules/RAFT/models/raft-small.pth"
MASK_THRESHOLD=1
MASK_KERNEL=3

mkdir -p "$WORK_DIR" "$ASSET_DIR"

echo "==============================================================="
echo "[Step 1/4] RAFT 前景/背景分离"
source "${conda_path}/bin/activate" nerfstudio

echo "==============================================================="
echo "==============================================================="
# python "$SCRIPT_DIR/extract_foreground_masks.py" \
#     --video "$VIDEO_PATH" \
#     --output "$WORK_DIR" \
#     --width $WIDTH --height $HEIGHT \
#     --mask_method raft \
#     --raft_model "$RAFT_MODEL" \
#     --mask_threshold "$MASK_THRESHOLD" \
#     --mask_kernel "$MASK_KERNEL" \
#     --save_foreground \
#     --max_frames "$MAX_FRAMES" \
#     --frame_interval "$FRAME_INTERVAL"

echo "==============================================================="
echo "[Step 2/4] 生成背景 MPI atlas"
echo "==============================================================="
source "${conda_path}/bin/activate" panosynthvr-py39
# python "$SCRIPT_DIR/generate_background_atlas.py" \
#     --frames_dir "$WORK_DIR/background_frames" \
#     --output "$WORK_DIR/background_atlas" \
#     --width 1024 --height 512 \
#     --max_frames "$MAX_FRAMES" \
#     --manifest "$WORK_DIR/background_manifest.json"

echo "==============================================================="
echo "[Step 3/4] 生成前景 MPI atlas"
echo "==============================================================="
# python "$SCRIPT_DIR/generate_foreground_atlas.py" \
#     --input_dir "$WORK_DIR/foreground_rgba" \
#     --output "$WORK_DIR/foreground_atlas" \
#     --width 1024 --height 512 \
#     --max_frames "$MAX_FRAMES" \
#     --manifest "$WORK_DIR/foreground_manifest.json"

echo "==============================================================="
echo "[Step 4/4] 拷贝结果到 docs/assets/${SCENE_NAME}"
echo "==============================================================="
rm -rf "$ASSET_DIR"
mkdir -p "$ASSET_DIR"
cp -r "$WORK_DIR/background_atlas"/frame_* "$ASSET_DIR"/
cp "$WORK_DIR/background_manifest.json" "$ASSET_DIR/frames_manifest.json"
mkdir -p "$ASSET_DIR/foreground_atlas"
cp -r "$WORK_DIR/foreground_atlas"/frame_* "$ASSET_DIR/foreground_atlas"/
cp "$WORK_DIR/foreground_manifest.json" "$ASSET_DIR/foreground_manifest.json"

echo "==============================================================="
echo "Pipeline 完成"
echo "==============================================================="
echo "使用浏览器访问:"
echo "  http://127.0.0.1:3600/docs/atlas_sequence_viewer.html?"
echo "     scene=${SCENE_NAME}&manifest=frames_manifest.json&foreground_manifest=foreground_manifest.json"
echo "可选参数: foreground_depth=1.02, bg_static=true, frame_step=2 等"
