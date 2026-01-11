#!/bin/bash

# ================================================================
# Hybrid Pipeline (External Masks):
# 1. 视频 -> Temporal Median 生成静态背景
# 2. 使用外部 mask PNG 序列生成 foreground_rgba（不跑 RAFT）
# 3. 背景: 静态背景生成 MPI atlas
# 4. 前景: foreground_rgba 生成动态 MPI atlas 序列
# 5. 自动部署并生成预览链接 (renderer.html?mode=hybrid)
#
# 用法:
#   ./scripts/run_hybrid_masks.sh <video_path|frames_dir> <mask_dir> <scene_name> [max_frames] [mask_glob] [mask_offset]
# 例子:
#   ./scripts/run_hybrid_masks.sh campus360.mp4 data/mask_birds/sam3_masks_pan_bird campus_sam 100 '*_obj000.png' 0
#   ./scripts/run_hybrid_masks.sh data/campus360_2048x1024_frames_200 data/mask_birds/sam3_masks_pan_bird campus_sam 100 '*_obj000.png' 0
# 说明:
#   - 默认分辨率固定为 2048x1024（可通过环境变量 WIDTH/HEIGHT 覆盖）
#   - 默认使用 GPU（可通过 FORCE_CPU=1 强制全程 CPU）
#   - 若 Step1 (median) 太慢：可用 BG_MODE=first 或降低 SAMPLE_COUNT
# ================================================================

set -euo pipefail

# 路径配置
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# 记录开始时间
START_TIME=$(date +%s)

# 参数解析
INPUT_SOURCE=${1:-}
MASK_DIR=${2:-}
BASE_SCENE_NAME=${3:-hybrid_masks_scene}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCENE_NAME="${BASE_SCENE_NAME}_${TIMESTAMP}"
MAX_FRAMES=${4:--1}
MASK_GLOB=${5:-"*.png"}
MASK_OFFSET=${6:-0}

if [[ -z "$INPUT_SOURCE" || -z "$MASK_DIR" ]]; then
    echo "Usage: $0 <video_path|frames_dir> <mask_dir> <scene_name> [max_frames] [mask_glob] [mask_offset]"
    echo "Example: $0 campus360.mp4 data/mask_birds/sam3_masks_pan_bird campus_sam 100 '*_obj000.png' 0"
    echo "Note: Output will be saved to .../${SCENE_NAME}"
    exit 1
fi

# Basic validation to avoid common arg-order mistakes.
if ! [[ "$MAX_FRAMES" =~ ^-?[0-9]+$ ]]; then
    echo "Error: max_frames must be an integer, got: $MAX_FRAMES"
    echo "Tip: arg order is [max_frames] [mask_glob] [mask_offset], e.g.:"
    echo "  $0 campus360.mp4 data/mask_birds/sam3_masks_pan_bird_1600x800 campus_sam 100 '*_obj000.png' 0"
    exit 1
fi
if ! [[ "$MASK_OFFSET" =~ ^-?[0-9]+$ ]]; then
    echo "Error: mask_offset must be an integer, got: $MASK_OFFSET"
    exit 1
fi

# ----------- 配置参数 -----------
WIDTH=${WIDTH:-2048}
HEIGHT=${HEIGHT:-1024}
BG_MODE=${BG_MODE:-median}           # median | first
SAMPLE_COUNT=${SAMPLE_COUNT:-100}    # 仅 BG_MODE=median 时生效

# 默认使用 GPU；设置 FORCE_CPU=1 强制全程 CPU
FORCE_CPU=${FORCE_CPU:-0}
if [[ "$FORCE_CPU" == "1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
    export TF_ENABLE_ONEDNN_OPTS=0
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
    export TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-1}
    export TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-1}
fi

WORK_DIR="$PROJECT_ROOT/output/hybrid_output/${SCENE_NAME}"
ASSET_DIR="$PROJECT_ROOT/docs/assets/${SCENE_NAME}"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-$CONDA_BASE}"
ENV_PREPROC="${ENV_PREPROC:-panosynthvr}"
ENV_TF="${ENV_TF:-panosynthvr311}"

activate_env() {
    local env_name="$1"
    if [[ -f "$CONDA_ACTIVATE/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_ACTIVATE/bin/activate" "$env_name" || true
    fi
}

# 清理旧数据
rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

echo "==============================================================="
echo "[Step 1/5] 生成静态背景 (Temporal Median)"
echo "==============================================================="
# 激活预处理环境
activate_env "$ENV_PREPROC"

TEMP_BG_DIR="$WORK_DIR/temp_static_bg"
mkdir -p "$TEMP_BG_DIR"
CLEAN_BG_PATH="$TEMP_BG_DIR/frame_000000.png"

if [[ "$BG_MODE" == "first" ]]; then
    python - <<PY
import cv2
import os
src = r"$INPUT_SOURCE"
if os.path.isdir(src):
    import glob
    paths = sorted(glob.glob(os.path.join(src, "frame_*.png")))
    if not paths:
        raise SystemExit("Error: No frame_*.png found for BG_MODE=first")
    frame = cv2.imread(paths[0], cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit("Error: Unable to read first frame from frames_dir")
else:
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Error: Unable to read first frame from video")
frame = cv2.resize(frame, (${WIDTH}, ${HEIGHT}), interpolation=cv2.INTER_AREA)
ok = cv2.imwrite(r"$CLEAN_BG_PATH", frame)
if not ok:
    raise SystemExit("Error: Failed to write clean background image")
print("Saved first-frame background to:", r"$CLEAN_BG_PATH")
PY
else
    if [[ -d "$INPUT_SOURCE" ]]; then
      python "$PROJECT_ROOT/main.py" clean_bg \
        --frames_dir "$INPUT_SOURCE" \
        --output "$CLEAN_BG_PATH" \
        --width "$WIDTH" --height "$HEIGHT" \
        --sample_count "$SAMPLE_COUNT"
    else
      python "$PROJECT_ROOT/main.py" clean_bg \
      --video "$INPUT_SOURCE" \
      --output "$CLEAN_BG_PATH" \
      --width "$WIDTH" --height "$HEIGHT" \
      --sample_count "$SAMPLE_COUNT"
    fi
fi

if [[ ! -f "$CLEAN_BG_PATH" ]]; then
    echo "Error: 无法生成干净背景 $CLEAN_BG_PATH"
    exit 1
fi

echo "==============================================================="
echo "[Step 2/5] 使用外部 Mask 提取前景 (RGBA)"
echo "==============================================================="
if [[ -d "$INPUT_SOURCE" ]]; then
    python "$PROJECT_ROOT/main.py" extract_masks \
      --frames_dir "$INPUT_SOURCE" \
      --mask_dir "$MASK_DIR" \
      --mask_glob "$MASK_GLOB" \
      --mask_offset "$MASK_OFFSET" \
      --output "$WORK_DIR" \
      --width "$WIDTH" --height "$HEIGHT" \
      --max_frames "$MAX_FRAMES" \
      --save_foreground
else
    python "$PROJECT_ROOT/main.py" extract_masks \
      --video "$INPUT_SOURCE" \
      --mask_dir "$MASK_DIR" \
      --mask_glob "$MASK_GLOB" \
      --mask_offset "$MASK_OFFSET" \
      --output "$WORK_DIR" \
      --width "$WIDTH" --height "$HEIGHT" \
      --max_frames "$MAX_FRAMES" \
      --save_foreground
fi

echo "==============================================================="
echo "[Step 3/5] 生成静态背景 MPI"
echo "==============================================================="
# 激活 TF/MPI 环境
activate_env "$ENV_TF"

python "$PROJECT_ROOT/main.py" background \
  --frames_dir "$TEMP_BG_DIR" \
  --output "$WORK_DIR/background_atlas" \
  --width "$WIDTH" --height "$HEIGHT" \
  --manifest "$WORK_DIR/background_manifest.json"

echo "==============================================================="
echo "[Step 4/5] 生成动态前景 MPI"
echo "==============================================================="
python "$PROJECT_ROOT/main.py" foreground \
  --input_dir "$WORK_DIR/foreground_rgba" \
  --output "$WORK_DIR/foreground_atlas" \
  --width "$WIDTH" --height "$HEIGHT" \
  --manifest "$WORK_DIR/foreground_manifest.json"

# 修正前景 Manifest 的 basePath
if [[ -f "$WORK_DIR/foreground_manifest.json" ]]; then
  sed -i 's/"basePath": ""/"basePath": "foreground_atlas\/"/' "$WORK_DIR/foreground_manifest.json"
fi

echo "==============================================================="
echo "[Step 5/5] 部署与索引更新"
echo "==============================================================="
rm -rf "$ASSET_DIR"
mkdir -p "$ASSET_DIR"

cp -r "$WORK_DIR/background_atlas"/frame_* "$ASSET_DIR"/
cp "$WORK_DIR/background_manifest.json" "$ASSET_DIR/frames_manifest.json"

mkdir -p "$ASSET_DIR/foreground_atlas"
cp -r "$WORK_DIR/foreground_atlas"/frame_* "$ASSET_DIR/foreground_atlas"/
cp "$WORK_DIR/foreground_manifest.json" "$ASSET_DIR/foreground_manifest.json"

# 更新索引页（用预处理环境即可）
activate_env "$ENV_PREPROC"
python "$PROJECT_ROOT/main.py" index

# 统计实际帧数
ACTUAL_FRAMES=$(ls "$ASSET_DIR/foreground_atlas" 2>/dev/null | wc -l)

# 记录结束时间并计算耗时
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

TIMESTAMP_URL=$(date +%s)
echo ""
echo "==============================================================="
echo "Hybrid Pipeline (Masks) 完成!"
echo "总耗时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "场景名称: $SCENE_NAME"
echo "实际生成帧数: $ACTUAL_FRAMES"
echo ""
echo "访问地址: http://127.0.0.1:3600/docs/renderer.html?mode=hybrid&name=$SCENE_NAME&frames=$ACTUAL_FRAMES&fps=24&t=$TIMESTAMP_URL"
echo "==============================================================="
