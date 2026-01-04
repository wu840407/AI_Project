import os

# ==========================================
# 1. 設定模型儲存路徑 (一定要在 import torch 之前)
# ==========================================
# 這行會讓 Hugging Face 把模型下載到您專案底下的 models_cache 資料夾
os.environ["HF_HOME"] = os.path.abspath("./models_cache")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from tqdm import tqdm
import time

# --- 設定輸入與輸出資料夾 ---
INPUT_FOLDER = "./input_audio"
OUTPUT_FOLDER = "./output_text"

# 支援的音檔格式
SUPPORTED_EXTENSIONS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')

# ==========================================
# 2. 載入模型 (自動下載到 D 槽)
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 正在初始化系統，使用裝置: {device}")
print(f"📂 模型儲存路徑: {os.environ['HF_HOME']}")

print("⏳ [1/2] 正在載入 Whisper-Large-v3 (負責聽)...")
# Whisper 會自動下載到 models_cache
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)

print("⏳ [2/2] 正在載入 Qwen-2.5-7B (負責翻譯)...")
llm_model_id = "Qwen/Qwen2.5-7B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# ==========================================
# 3. 定義處理邏輯
# ==========================================
def process_single_file(file_path, output_path):
    start_time = time.time()
    filename = os.path.basename(file_path)
    
    # --- A. Whisper 自動識別 ---
    # 我們不指定 language，讓 Whisper 自己猜 (它支援 auto detect)
    # return_timestamps=True 讓它處理長音檔更穩定
    try:
        asr_output = asr_pipe(
            file_path, 
            generate_kwargs={"task": "transcribe"}, # transcribe = 轉錄原文
            return_timestamps=True
        )
        raw_text = asr_output["text"]
    except Exception as e:
        print(f"❌ Whisper 識別失敗: {filename}, 錯誤: {e}")
        return

    # --- B. LLM 翻譯與方言識別 ---
    # 這裡的 Prompt 是關鍵，我們讓 Qwen 自己去判斷原文是哪種方言
    system_prompt = """
    你是一位精通漢語方言（四川話、上海話、廣東話、閩南語）以及維吾爾語的語言學家。
    
    使用者的輸入是一段語音識別（ASR）後的文字。
    你的任務是：
    1. 【判斷語言】：分析這段文字屬於哪種語言或方言。
    2. 【翻譯】：將其準確翻譯為「標準正體中文」。
    3. 【輸出格式】：請嚴格依照下方格式輸出，不要有多餘廢話。
    
    格式範例：
    [語言]: 四川話
    [原文]: (這裡放原本識別的文字)
    [譯文]: (這裡放正體中文翻譯)
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"請處理這段識別文字：\n{raw_text}"}
    ]

    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    # 產生翻譯
    with torch.no_grad():
        generated_ids = llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            temperature=0.3 # 翻譯需要準確度，溫度調低
        )
    
    final_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # 移除 Prompt 部分，只保留 AI 回答 (Qwen有時會包含 prompt，視版本而定，通常 skip_special_tokens 就夠了，這裡做字串處理保險)
    # 這裡假設 Qwen 直接輸出回答。若有包含 input，通常在 tokenizer decode 時會處理，或是用 output_ids[len(input_ids):] 切割
    # 為了程式碼簡潔，這裡使用簡單的切割法（如果模型輸出了 prompt）
    if "請處理這段識別文字" in final_output:
         final_output = final_output.split("請處理這段識別文字：")[-1].strip()

    # --- C. 存檔 ---
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output)

    duration = time.time() - start_time
    print(f"✅ 完成: {filename} (耗時 {duration:.2f}秒)")

# ==========================================
# 4. 主程式執行
# ==========================================
if __name__ == "__main__":
    # 確保輸出資料夾存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 掃描檔案
    all_files = [
        f for f in os.listdir(INPUT_FOLDER) 
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    
    print(f"\n📂 在 {INPUT_FOLDER} 找到 {len(all_files)} 個音檔，準備開始處理...\n")
    
    # 使用 tqdm 顯示進度條
    for filename in tqdm(all_files, desc="總進度"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + "_translated.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        process_single_file(input_path, output_path)

    print(f"\n🎉 全部完成！請查看 {OUTPUT_FOLDER} 資料夾。")