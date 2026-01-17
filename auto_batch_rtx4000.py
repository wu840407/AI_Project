import os
import sys
import torch
import time
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# ==========================================
# 0. 硬體與路徑配置 (Server 版核心設定)
# ==========================================
# 設定 Hugging Face 快取 (建議指向大硬碟)
os.environ["HF_HOME"] = "/data/models_cache" 

# 設定輸入與輸出資料夾 (直接指向 RAID 10 大硬碟 /data)
# 您需要先在 /data 建立這些資料夾: mkdir -p /data/input_audio /data/output_text
INPUT_FOLDER = "/data/input_audio"
OUTPUT_FOLDER = "/data/output_text"

# 支援的音檔格式
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')

# 離線模型路徑 (如果有的話，優先讀取這裡)
OFFLINE_MODEL_PATH_LLM = "/data/ai_models/Llama-3.1-8B-Instruct"
# 如果沒有離線檔，就用網路 ID
ONLINE_MODEL_ID_LLM = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --- 雙卡分配邏輯 ---
if torch.cuda.device_count() >= 2:
    print(f"🚀 [Server Mode] 偵測到雙顯卡，啟動平行加速！")
    DEVICE_ASR = "cuda:0"  # GPU 0 聽
    DEVICE_LLM = "cuda:1"  # GPU 1 想
else:
    print(f"⚠️ [Single Mode] 僅偵測到單卡，使用混合模式。")
    DEVICE_ASR = "cuda:0"
    DEVICE_LLM = "cuda:0"

print(f"🎤 ASR Device: {DEVICE_ASR}")
print(f"🧠 LLM Device: {DEVICE_LLM}")

# ==========================================
# 1. 載入模型
# ==========================================

# --- A. 載入 Whisper (GPU 0) ---
print("⏳ [1/2] 正在 GPU 0 載入 Whisper-Large-v3...")
try:
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3", # 如果有離線版可改成路徑
        torch_dtype=torch.float16,
        device=DEVICE_ASR,
    )
except Exception as e:
    print(f"❌ Whisper 載入失敗: {e}")
    sys.exit(1)

# --- B. 載入 Llama 3 (GPU 1) ---
# 判斷要用離線路徑還是線上 ID
if os.path.exists(OFFLINE_MODEL_PATH_LLM):
    print(f"⏳ [2/2] 正在 GPU 1 載入 Llama 3.1 (離線版: {OFFLINE_MODEL_PATH_LLM})...")
    model_source = OFFLINE_MODEL_PATH_LLM
else:
    print(f"⏳ [2/2] 正在 GPU 1 載入 Llama 3.1 (線上下載: {ONLINE_MODEL_ID_LLM})...")
    model_source = ONLINE_MODEL_ID_LLM

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

try:
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_source,
        quantization_config=bnb_config,
        device_map=DEVICE_LLM,  # <--- 強制指定到 GPU 1
        trust_remote_code=True
    )
except Exception as e:
    print(f"❌ LLM 載入失敗: {e}")
    print("提示: 若使用線上 ID，請確認已執行 `huggingface-cli login`")
    sys.exit(1)

# ==========================================
# 2. 定義處理邏輯
# ==========================================
def process_single_file(file_path, output_path):
    start_time = time.time()
    filename = os.path.basename(file_path)
    
    # --- A. 前處理：PSTN 優化 ---
    try:
        # 強制轉為 16000Hz (Whisper 原生頻率)
        y, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"❌ 讀取失敗: {filename}, 錯誤: {e}")
        return

    # 正規化 (Normalization)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    # --- B. Whisper 識別 (GPU 0) ---
    try:
        asr_output = asr_pipe(
            {"raw": y, "sampling_rate": 16000}, 
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )
        raw_text = asr_output["text"]
    except Exception as e:
        print(f"❌ Whisper 識別失敗: {filename}, 錯誤: {e}")
        return

    # --- C. LLM 分析 (GPU 1) ---
    # 使用 Llama 3 的 Prompt 格式
    system_prompt = """
    You are a linguistic expert specializing in Chinese dialects and Uyghur.
    The input text comes from phone recordings (ASR output) and may contain errors.
    
    Your task:
    1. [Identify]: Detect the language/dialect (e.g., Sichuan, Cantonese, Uyghur).
    2. [Translate]: Translate it into standard Traditional Chinese (正體中文).
    3. [Correct]: Fix homophone errors based on context (e.g., "四" vs "十").
    
    Output Format:
    [語言]: (Detected Language)
    [原文]: (Original ASR text)
    [譯文]: (Translated Text)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please process this text:\n{raw_text}"}
    ]

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 注意：將輸入搬移到 DEVICE_LLM (GPU 1)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(DEVICE_LLM)

    with torch.no_grad():
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    final_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # --- D. 存檔 ---
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    duration = time.time() - start_time
    # 簡單進度顯示
    tqdm.write(f"✅ 完成: {filename} ({duration:.1f}s)")

# ==========================================
# 3. 主程式執行
# ==========================================
if __name__ == "__main__":
    # 確保資料夾存在
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ 建立輸入資料夾: {INPUT_FOLDER}")
        print(f"👉 請將音檔放入此資料夾後重新執行。")
        sys.exit(0)
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 掃描檔案
    all_files = [
        f for f in os.listdir(INPUT_FOLDER) 
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    
    if len(all_files) == 0:
        print(f"📂 {INPUT_FOLDER} 是空的。請放入音檔。")
        sys.exit(0)

    print(f"\n📂 來源: {INPUT_FOLDER}")
    print(f"📂 目的: {OUTPUT_FOLDER}")
    print(f"📊 總計: {len(all_files)} 個檔案\n")
    
    # 批次處理
    for filename in tqdm(all_files, desc="Processing"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + "_report.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        process_single_file(input_path, output_path)

    print(f"\n🎉 全部完成！")