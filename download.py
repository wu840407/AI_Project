import os
import shutil
from modelscope import snapshot_download

TARGET_DIR = "/data/ai_models"

# 1. 刪除被 404 網頁污染的資料夾
corrupted_dir = os.path.join(TARGET_DIR, "iic/SenseVoiceSmall")
if os.path.exists(corrupted_dir):
    shutil.rmtree(corrupted_dir)
    print("🗑️ 已清除受污染的 SenseVoice 資料夾")

# 2. 重新下載純淨版 SenseVoice (不含 model.py 是正常的！)
print("⏳ [1/2] 重新下載純淨版 SenseVoiceSmall...")
snapshot_download('iic/SenseVoiceSmall', cache_dir=TARGET_DIR, revision='master')

# 3. 下載 VAD 模型 (長音檔切片神器，這是準確率的關鍵)
print("⏳ [2/2] 下載 FSMN-VAD 語音端點偵測模型...")
snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir=TARGET_DIR, revision='master')

print("✅ 重建完畢！您的本地彈藥庫已就緒。")
EOF

