#!/bin/bash

# ================================================================
# Video Atlas Pipeline
# 1. 输入视频 -> 为每一帧生成 MPI atlas
# 2. 将 atlas.png 序列编码成 RGB|Alpha 双拼视频 atlas_video.mp4
# 3. 拷贝到 docs/assets/<scene>/ 供 renderer.html?mode=video_atlas 使用
#
# 用法:
#   ./scripts/run_video_atlas_pipeline.sh <video_path> <scene_name> [width] [height] [fps] [max_frames]
# 例如:
#   ./scripts/run_video_atlas_pipeline.sh assets/campus360.mp4 campus_video 1024 512 24 180
# ================================================================

set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

VIDEO_PATH=${1:-}
SCENE_NAME=${2:-video_atlas_scene}
WIDTH=${3:-1024}
HEIGHT=${4:-512}
FPS=${5:-24}
MAX_FRAMES=${6:--1}

if [[ -z "$VIDEO_PATH" ]]; then
    echo "Usage: $0 <video_path> <scene_name> [width] [height] [fps] [max_frames]"
    exit 1
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SCENE_OUTPUT_NAME="${SCENE_NAME}_${TIMESTAMP}"
WORK_DIR="$PROJECT_ROOT/output/video_atlas/${SCENE_OUTPUT_NAME}"
ATLAS_DIR="$WORK_DIR/frames"
OUTPUT_VIDEO="$WORK_DIR/atlas_video.mp4"
ASSET_DIR="$PROJECT_ROOT/docs/assets/${SCENE_OUTPUT_NAME}"
CONDA_ACTIVATE="/home/ksi/miniconda3"

rm -rf "$WORK_DIR"
mkdir -p "$ATLAS_DIR"

# 开始计时
PIPELINE_START=$(date +%s)

echo "==============================================================="
echo "[1/3] 生成逐帧 Atlas"
echo "==============================================================="
STEP1_START=$(date +%s)
if [[ -d "$CONDA_ACTIVATE" && -f "$CONDA_ACTIVATE/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_ACTIVATE"/bin/activate panosynthvr-py39
fi
python "$PROJECT_ROOT/scripts/generate_full_atlas_video.py" \
    --input "$VIDEO_PATH" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --output "$ATLAS_DIR" \
    --max_frames "$MAX_FRAMES"

STEP1_END=$(date +%s)
STEP1_DURATION=$((STEP1_END - STEP1_START))
echo "✓ Step 1 completed in ${STEP1_DURATION}s"

echo "==============================================================="
echo "[2/3] 预处理双拼帧并使用 FFmpeg 打包"
echo "==============================================================="
STEP2_START=$(date +%s)
RGB_FRAMES_DIR="$WORK_DIR/rgb_frames"
ALPHA_FRAMES_DIR="$WORK_DIR/alpha_frames"
rm -rf "$RGB_FRAMES_DIR" "$ALPHA_FRAMES_DIR"
mkdir -p "$RGB_FRAMES_DIR" "$ALPHA_FRAMES_DIR"

ATLAS_DIR_ENV="$ATLAS_DIR" RGB_DIR_ENV="$RGB_FRAMES_DIR" ALPHA_DIR_ENV="$ALPHA_FRAMES_DIR" MAX_FRAMES_ENV="$MAX_FRAMES" python - <<'PY'
import cv2, numpy as np, pathlib, os

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

frames_dir = pathlib.Path(os.environ["ATLAS_DIR_ENV"])
rgb_dir = pathlib.Path(os.environ["RGB_DIR_ENV"])
alpha_dir = pathlib.Path(os.environ["ALPHA_DIR_ENV"])
max_frames = int(os.environ["MAX_FRAMES_ENV"])

frame_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")])

count = 0
iterator = tqdm(frame_dirs, desc="Preparing RGB/Alpha PNGs") if tqdm else frame_dirs
for frame_dir in iterator:
    atlas_path = frame_dir / "atlas.png"
    if not atlas_path.exists():
        continue
    img = cv2.imread(str(atlas_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue

    if img.shape[2] == 4:
        color = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        alpha = img[:, :, 3]
    else:
        color = img[:, :, :3] if img.shape[2] >= 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        alpha = np.full(color.shape[:2], 255, dtype=np.uint8)

    alpha_rgb = cv2.merge([alpha, alpha, alpha])
    cv2.imwrite(str(rgb_dir / f"rgb_{count:06d}.png"), color)
    cv2.imwrite(str(alpha_dir / f"alpha_{count:06d}.png"), alpha_rgb)
    count += 1
    if max_frames > -1 and count >= max_frames:
        break

print(f"Prepared {count} color/alpha frame pairs")
PY

FFMPEG_BIN=${FFMPEG_BIN:-ffmpeg}
FFMPEG_THREADS=${FFMPEG_THREADS:-8}
FFMPEG_PRESET=${FFMPEG_PRESET:-faster}
FFMPEG_CRF=${FFMPEG_CRF:-18}
RGB_VIDEO="$WORK_DIR/atlas_rgb.mp4"
ALPHA_VIDEO="$WORK_DIR/atlas_alpha.mp4"

$FFMPEG_BIN -y -framerate "$FPS" -i "$RGB_FRAMES_DIR/rgb_%06d.png" \
    -c:v libx264 -preset "$FFMPEG_PRESET" -crf "$FFMPEG_CRF" \
    -threads "$FFMPEG_THREADS" -pix_fmt yuv420p "$RGB_VIDEO"

$FFMPEG_BIN -y -framerate "$FPS" -i "$ALPHA_FRAMES_DIR/alpha_%06d.png" \
    -c:v libx264 -preset "$FFMPEG_PRESET" -crf "$FFMPEG_CRF" \
    -threads "$FFMPEG_THREADS" -pix_fmt yuv420p "$ALPHA_VIDEO"

STEP2_END=$(date +%s)
STEP2_DURATION=$((STEP2_END - STEP2_START))
echo "✓ Step 2 completed in ${STEP2_DURATION}s"

echo "==============================================================="
echo "[3/3] 拷贝到 docs/assets/${SCENE_OUTPUT_NAME}"
echo "==============================================================="
STEP3_START=$(date +%s)
rm -rf "$ASSET_DIR"
mkdir -p "$ASSET_DIR"
cp "$RGB_VIDEO" "$ASSET_DIR/atlas_rgb.mp4"
cp "$ALPHA_VIDEO" "$ASSET_DIR/atlas_alpha.mp4"

# 复制第一帧用于调试（可选）
FIRST_FRAME_DIR=$(find "$ATLAS_DIR" -maxdepth 1 -type d -name "frame_*" | sort | head -n 1)
if [[ -n "$FIRST_FRAME_DIR" ]]; then
    cp -r "$FIRST_FRAME_DIR" "$ASSET_DIR/$(basename "$FIRST_FRAME_DIR")"
fi

STEP3_END=$(date +%s)
STEP3_DURATION=$((STEP3_END - STEP3_START))
echo "✓ Step 3 completed in ${STEP3_DURATION}s"

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
PIPELINE_MINUTES=$((PIPELINE_DURATION / 60))
PIPELINE_SECONDS=$((PIPELINE_DURATION % 60))

echo ""
echo "==============================================================="
echo "Video atlas generated!"
echo "==============================================================="
echo "Scene: $SCENE_OUTPUT_NAME"
echo "Timing:"
echo "  - Step 1 (Atlas generation): ${STEP1_DURATION}s"
echo "  - Step 2 (Video encoding):    ${STEP2_DURATION}s"
echo "  - Step 3 (Asset copy):        ${STEP3_DURATION}s"
echo "  - Total:                      ${PIPELINE_MINUTES}m ${PIPELINE_SECONDS}s"
echo ""
echo "Preview: http://127.0.0.1:3600/docs/renderer.html?mode=video_atlas&scene=-1&name=$SCENE_OUTPUT_NAME"
echo ""
