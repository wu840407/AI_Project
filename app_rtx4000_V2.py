import os
import sys
import torch
import math
import re
import gradio as gr
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    BitsAndBytesConfig
)

# ==========================================
# 0. 雙卡戰略配置 (Dual GPU Config)
# ==========================================
# 設定記憶體管理參數以避免碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/data/models_cache"

# 定義分工
DEVICE_ASR = "cuda:0"  # 第一張卡：負責聽 (Whisper + Pyannote)
DEVICE_LLM = "cuda:1"  # 第二張卡：負責想 (Llama)

print(f"🚀 啟動雅言 AI (V3.1 雙卡戰略版)...")
print(f"   - 耳朵 (ASR) 配置於: {DEVICE_ASR}")
print(f"   - 大腦 (LLM) 配置於: {DEVICE_LLM}")

# 嘗試匯入 Pyannote
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("⚠️ 未偵測到 pyannote.audio，將無法執行 A/B 方聲紋區分。")

# 模型路徑
OFFLINE_MODEL_PATH_LLM = "/data/ai_models/Llama-3.1-8B-Instruct"
OFFLINE_MODEL_PATH_WHISPER = "/data/ai_models/whisper-large-v3" 
LOCAL_PYANNOTE_PATH = "/data/ai_models/pyannote-speaker-diarization-3.1/config.yaml"

# ==========================================
# 1. 載入模型 (分開載入)
# ==========================================

# A. Whisper (載入到 GPU 0)
print(f"⏳ [1/3] 載入 Whisper (on {DEVICE_ASR})...")
try:
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=OFFLINE_MODEL_PATH_WHISPER, 
        model_kwargs={"dtype": torch.float16}, 
        device=DEVICE_ASR,  # <--- 指定 GPU 0
    )
except Exception as e:
    print(f"❌ Whisper 載入失敗: {e}")
    sys.exit(1)

# B. Llama (載入到 GPU 1)
print(f"⏳ [2/3] 載入 Llama 3.1 (on {DEVICE_LLM})...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
llm_model = AutoModelForCausalLM.from_pretrained(
    OFFLINE_MODEL_PATH_LLM,
    quantization_config=bnb_config, 
    device_map={"":"cuda:1"}, # <--- 強制指定 GPU 1
    trust_remote_code=True,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(OFFLINE_MODEL_PATH_LLM)
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

# C. Pyannote (載入到 GPU 0，與 Whisper 共用)
diarization_pipeline = None
if PYANNOTE_AVAILABLE:
    print(f"⏳ [3/3] 載入聲紋辨識 (on {DEVICE_ASR})...")
    try:
        if os.path.exists(LOCAL_PYANNOTE_PATH):
            diarization_pipeline = Pipeline.from_pretrained(LOCAL_PYANNOTE_PATH)
        else:
            diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
            
        diarization_pipeline.to(torch.device(DEVICE_ASR)) # <--- 指定 GPU 0
    except Exception as e:
        print(f"⚠️ 聲紋模型載入失敗: {e}")
        diarization_pipeline = None

# ==========================================
# 2. 核心功能函式 (保持不變)
# ==========================================
# ... (以下這段不用改，照抄原本 V3 的函式即可) ...

def filter_hallucination(text, avg_logprob=None):
    blacklist = ["請訂閱", "點讚", "開啟小鈴鐺", "字幕", "Subtitle", "Amara.org", "Thank you", "watching", "Copyright", "MBC", "SBS", "訂閱頻道"]
    for word in blacklist:
        if word in text: return "(無聲/背景音)"
    if avg_logprob is not None and avg_logprob < -1.0 and len(text) < 5: return "(無聲/背景音)"
    if len(text) > 10 and len(set(text)) < 4: return "(無聲/雜訊)"
    return text

def format_time(seconds):
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{int(s):02d}"

def process_audio(audio_path, source_dialect, analysis_mode):
    if audio_path is None: return "無音訊", "", ""
    print(f"🎤 收到任務 | 語言: {source_dialect} | 模式: {analysis_mode}")
    
    # --- 步驟 A: 聲紋 (GPU 0) ---
    diarization_result = []
    if diarization_pipeline:
        print("🔍 A/B 方聲紋分析中...")
        try:
            dia = diarization_pipeline(audio_path)
            for turn, _, speaker in dia.itertracks(yield_label=True):
                diarization_result.append((turn.start, turn.end, speaker))
        except Exception as e:
            print(f"聲紋分析錯誤 (略過): {e}")
    
    # --- 步驟 B: Whisper (GPU 0) ---
    print("📝 Whisper 聽寫中...")
    asr_result = asr_pipe(
        audio_path, 
        chunk_length_s=30,
        batch_size=8,
        generate_kwargs={"task": "transcribe", "language": "chinese"},
        return_timestamps=True
    )
    
    # --- 步驟 C: 整合 ---
    final_transcript = []
    whisper_chunks = asr_result.get("chunks", [])
    full_raw_text = ""
    all_logprobs = []
    for chunk in whisper_chunks:
        start, end = chunk['timestamp']
        text = chunk['text']
        avg_logprob = chunk.get('avg_logprob', -1.0)
        all_logprobs.append(avg_logprob)
        filtered_text = filter_hallucination(text, avg_logprob)
        if filtered_text == "(無聲/背景音)": continue 
        mid_time = (start + end) / 2
        speaker_label = "未知"
        if diarization_result:
            for d_start, d_end, d_spk in diarization_result:
                if d_start <= mid_time <= d_end:
                    speaker_label = d_spk.replace("SPEAKER_00", "A方").replace("SPEAKER_01", "B方")
                    break
        line = f"[{format_time(start)}] {speaker_label}: {filtered_text}"
        final_transcript.append(line)
        full_raw_text += line + "\n"
    if not final_transcript: full_raw_text = asr_result['text']

    # --- 步驟 D: 信心分數 ---
    confidence_score = 0.0
    if all_logprobs:
        try:
            probability = math.exp(sum(all_logprobs) / len(all_logprobs))
        except: probability = 0.0
        confidence_score = round(probability * 100, 1)
    
    if confidence_score > 80: color = "green"
    elif confidence_score > 60: color = "orange"
    else: color = "red"
    confidence_html = f"<div style='text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px; background-color: #fafafa;'><div style='font-size: 14px; color: gray;'>AI 聽寫信心指數 (LogProb)</div><div style='font-size: 36px; font-weight: bold; color: {color};'>{confidence_score}%</div><div style='font-size: 12px; color: gray;'>若低於 60% 代表方言辨識困難</div></div>"

    # --- 步驟 E: Llama (GPU 1) ---
    print(f"🧠 Llama 正在進行: {analysis_mode}...")
    
    dialect_prompt_map = {
        "台灣口語/台語": "注意：對話包含台語（閩南語）。Whisper 可能將其轉錄為同音的國語字（如 '哩賀'->'你好'，'瓦'->'我'）。請根據語境自動修正並翻譯。",
        "廣東話 (粵語)": "注意：對話為廣東話（粵語）。逐字稿可能包含粵語語法（如 '係'、'咁'、'嘅'）或同音錯字。請將其視為粵語並進行語意理解。",
        "四川話": "注意：對話為四川方言。請注意四川話特有的詞彙（如 '耍'->'玩'，'曉得'->'知道'）及語氣助詞，修正辨識錯誤。",
        "上海話 (吳語)": "注意：對話為上海話（吳語）。Whisper 對吳語的辨識能力較弱，逐字稿可能充滿同音錯字。請極力根據上下文推斷原意。",
        "山東話": "注意：對話為山東方言。請注意倒裝句及特有語氣（如 '俺'->'我'），修正可能的語意誤判。",
        "英語": "Note: The source audio is in English. Please analyze the content in Traditional Chinese."
    }
    dialect_instruction = dialect_prompt_map.get(source_dialect, "注意：請修正語音辨識可能產生的同音異字錯誤。")

    mode_instruction = ""
    if analysis_mode == "總結分析":
        mode_instruction = "請提供【標準情報摘要】：1. 對話主題。2. 重點摘要。3. 待辦事項。"
    elif analysis_mode == "戰略意圖分析":
        mode_instruction = "請進行【戰略意圖研判】：1. A方目的與情緒。2. B方立場與防衛心理。3. 雙方權力與利益衝突研判。"
    elif analysis_mode == "謀略導變建議":
        mode_instruction = "請提供【謀略導變建議】：1. 局勢判斷(有利/不利)。2. 應對策略(引用孫子兵法/博弈論)。3. 話術指導。"

    system_prompt = f"你是一個高階情報分析 AI。任務是閱讀「語音辨識後的逐字稿」並進行分析。\n【方言指令】{dialect_instruction}\n【任務】{mode_instruction}\n請用專業、精簡的繁體中文輸出。"

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"逐字稿內容：\n{full_raw_text}"}]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 注意：這裡也要送到 GPU 1
    model_inputs = tokenizer([text_input], return_tensors="pt").to(DEVICE_LLM) 

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=1500,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(generated_ids[0][len(model_inputs.input_ids[0]):], skip_special_tokens=True)
    
    return confidence_html, full_raw_text, response

# ==========================================
# 3. Gradio 介面 (V3.1)
# ==========================================
with gr.Blocks(title="雅言 AI - 戰情儀表板", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏺 雅言 AI - 戰情儀表板 (V3.1 雙卡運算版)")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎛️ 任務控制台")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="上傳音檔")
            with gr.Row():
                dialect_dropdown = gr.Dropdown(choices=["台灣口語/台語", "廣東話 (粵語)", "四川話", "上海話 (吳語)", "山東話", "標準國語", "英語"], value="台灣口語/台語", label="🗣️ 來源語言")
                mode_dropdown = gr.Dropdown(choices=["總結分析", "戰略意圖分析", "謀略導變建議"], value="總結分析", label="🧠 研判模式")
            submit_btn = gr.Button("🚀 執行戰術分析", variant="primary", size="lg")
            confidence_out = gr.HTML()
        with gr.Column(scale=2):
            gr.Markdown("### 📊 情報研判結果")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📝 A/B 方聲紋逐字稿")
                    raw_out = gr.Textbox(label="Whisper + Pyannote 原始輸出", lines=20, show_label=False, interactive=False)
                with gr.Column():
                    gr.Markdown("#### 🧠 AI 深度研判")
                    analysis_out = gr.Textbox(label="Llama 3.1 分析結果", lines=20, show_label=False, interactive=False)
    submit_btn.click(fn=process_audio, inputs=[audio_input, dialect_dropdown, mode_dropdown], outputs=[confidence_out, raw_out, analysis_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
