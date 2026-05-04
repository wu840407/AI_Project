#!/usr/bin/env bash
# 啟動 YaYan-AI v4.5 伺服器（離線模式）
# 用法: bash scripts/start_server.sh
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export YAYAN_MODELS_ROOT="${YAYAN_MODELS_ROOT:-/data/ai_models}"
export YAYAN_INPUT_DIR="${YAYAN_INPUT_DIR:-/data/input_audio}"
export YAYAN_OUTPUT_DIR="${YAYAN_OUTPUT_DIR:-/data/output_text}"

mkdir -p "$YAYAN_INPUT_DIR" "$YAYAN_OUTPUT_DIR"

echo "============================================"
echo " YaYan-AI v4.5  Server (Offline Mode)"
echo " Models: $YAYAN_MODELS_ROOT"
echo " Input : $YAYAN_INPUT_DIR"
echo " Output: $YAYAN_OUTPUT_DIR"
echo "============================================"

python verify_offline.py 2>/dev/null || python scripts/verify_models.py

exec python app_rtx6000.py
