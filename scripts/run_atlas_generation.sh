#!/bin/bash
# 自动激活panosynthvr环境并运行脚本

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate panosynthvr

# 运行脚本，传递所有参数
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
python "$SCRIPT_DIR/generate_full_atlas_video.py" "$@"
