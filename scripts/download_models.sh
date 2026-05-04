#!/usr/bin/env bash
# 在「連網機器」執行，會把所有 YaYan-AI v4.5 需要的模型下載到 $MODELS_ROOT
# 之後再用 rsync / SCP 把整個 ai_models 資料夾搬到離線伺服器
#
# 用法:
#   bash download_models.sh                    # 下載核心模型
#   bash download_models.sh --with-diarize     # 額外下載 pyannote（需 HF token）
#   MODELS_ROOT=/data/ai_models bash download_models.sh

set -euo pipefail

MODELS_ROOT="${MODELS_ROOT:-/data/ai_models}"
WITH_DIARIZE=0
for arg in "$@"; do
  case "$arg" in
    --with-diarize) WITH_DIARIZE=1 ;;
  esac
done

mkdir -p "$MODELS_ROOT"

echo "============================================"
echo " YaYan-AI v4.5  模型下載"
echo " 目標路徑：$MODELS_ROOT"
echo "============================================"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "❌ 找不到 huggingface-cli，請先：pip install -U huggingface_hub"
  exit 1
fi

download_to() {
  # download_to <hf_repo_id> <local_dir>
  local repo="$1"
  local dest="$MODELS_ROOT/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null || true)" ]; then
    echo "⏩ 已存在，略過: $dest"
    return
  fi
  echo "⬇️  下載 $repo → $dest"
  huggingface-cli download "$repo" --local-dir "$dest" --local-dir-use-symlinks False
}

# === LLM（Translator/Reasoner 共用權重）===
download_to "Qwen/Qwen3-14B-Instruct" "YaYan_Reasoner"

# === 漢語方言 ASR ===
download_to "FunAudioLLM/SenseVoiceSmall" "YaYan_ASR_Mandarin"

# === 東方語族 ASR（藏/維/吳）===
download_to "DataoceanAI/dolphin-base" "YaYan_ASR_Eastern"

# === 通用 ASR（波斯/烏爾都/英）===
download_to "openai/whisper-large-v3" "YaYan_ASR_Global"

# === VAD ===
download_to "snakers4/silero-vad" "YaYan_VAD"

# === LID ===
download_to "speechbrain/lang-id-voxlingua107-ecapa" "YaYan_LID"

# === Diarization（gated，需 HF token + 同意條款）===
if [ "$WITH_DIARIZE" = "1" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "⚠️  需要 HF_TOKEN 才能下載 pyannote。請先："
    echo "    export HF_TOKEN=hf_xxx"
    echo "    並到 https://huggingface.co/pyannote/speaker-diarization-3.1 同意條款"
    exit 1
  fi
  HUGGINGFACE_HUB_TOKEN="$HF_TOKEN" \
    download_to "pyannote/speaker-diarization-3.1" "YaYan_Diarize"
fi

# === OpenCC 字典（pip 包自帶，無需額外下載）===

echo "============================================"
echo " ✅ 模型下載完成"
echo "    總大小："
du -sh "$MODELS_ROOT" || true
echo ""
echo " 下一步："
echo "   rsync -av --progress $MODELS_ROOT/ user@server:/data/ai_models/"
echo "============================================"
