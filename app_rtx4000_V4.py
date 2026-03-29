"""
雅言 AI - V4.1 方言語音對話轉寫系統
=====================================
架構：RTX 4000 × 2 雙卡優化版
- cuda:0 → Pyannote 3.1 (說話者分離) + FunASR SenseVoice (方言 ASR)
- cuda:1 → Whisper Large V3 (通用 ASR) + Qwen2.5-7B-Instruct (LLM 校正+分析)

V4.1 修正：
- FunASR 移除 trust_remote_code（本地模式不需要）
- Pyannote 4.x 移除已廢棄的 use_auth_token 參數
- 新增信心度儀表板顯示
- 新增 AI 對話修正框（可對輸出結果進行追問與修正）
"""

import os
import sys
import torch
import math
import re
import numpy as np
import soundfile as sf
import librosa
import gradio as gr
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from funasr import AutoModel
from pyannote.audio import Pipeline as PyannotePipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

# ==========================================
# 0. 雙卡環境配置
# ==========================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/data/models_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE_ASR   = "cuda:0"   # 耳朵：Pyannote + FunASR SenseVoice
DEVICE_WHISP = "cuda:1"   # 備耳：Whisper Large V3
DEVICE_LLM   = "cuda:1"   # 大腦：Qwen2.5-7B（與 Whisper 共用 cuda:1，交替使用）

print("🚀 啟動雅言 AI V4.0 (方言對話轉寫版)...")

# ==========================================
# 模型路徑設定（請依實際存放位置修改）
# ==========================================
MODEL_PATHS = {
    # 說話者分離
    "pyannote_diarization": "/data/ai_models/pyannote-speaker-diarization-3.1",
    "pyannote_segmentation": "/data/ai_models/pyannote-segmentation-3.0",

    # ASR
    "sensevoice":    "/data/ai_models/iic/SenseVoiceSmall",        # FunASR 會自動快取
    "whisper":       "/data/ai_models/whisper-large-v3",
    "whisper_tw":    "/data/ai_models/whisper-taiwanese1",  # 台語微調版（選用）

    # LLM
    "qwen":          "/data/ai_models/Qwen2.5-7B-Instruct",
}

# ==========================================
# 1. 載入模型
# ==========================================

# --- 1A: FunASR SenseVoice（方言 ASR，cuda:0）---
print(f"⏳ [1/4] 載入 FunASR SenseVoice on {DEVICE_ASR}...")
try:
    sense_voice_model = AutoModel(
        model=MODEL_PATHS["sensevoice"],
        device=DEVICE_ASR,
        disable_update=True,
        local_files_only=True
    )
    print("✅ FunASR SenseVoice 載入完成")
except Exception as e:
    print(f"❌ FunASR 載入失敗: {e}")
    sys.exit(1)

# --- 1B: Whisper Large V3（通用/外語 ASR，cuda:1）---
print(f"⏳ [2/4] 載入 Whisper V3 on {DEVICE_WHISP}...")
try:
    # 若有台語微調版且目標為台語，後面路由時會動態切換
    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_PATHS["whisper"],
        model_kwargs={"dtype": torch.float16},
        device=DEVICE_WHISP,
    )

    # 台語專屬微調版（若模型存在則載入）
    whisper_tw_pipe = None
    if Path(MODEL_PATHS["whisper_tw"]).exists():
        whisper_tw_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_PATHS["whisper_tw"],
            model_kwargs={"dtype": torch.float16},
            device=DEVICE_WHISP,
        )
        print("✅ Whisper V3 + 台語微調版 載入完成")
    else:
        print("✅ Whisper V3 載入完成（台語微調版未找到，使用通用版）")
except Exception as e:
    print(f"❌ Whisper 載入失敗: {e}")
    sys.exit(1)

# --- 1C: Qwen2.5-7B-Instruct（LLM 校正+分析，cuda:1）---
print(f"⏳ [3/4] 載入 Qwen2.5-7B on {DEVICE_LLM}...")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Qwen2.5 建議用 bfloat16
        bnb_4bit_use_double_quant=True,           # 節省約 0.4GB 顯存
    )
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATHS["qwen"],
        quantization_config=bnb_config,
        device_map={"": DEVICE_LLM},
        trust_remote_code=True,
        local_files_only=True,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS["qwen"],
        trust_remote_code=True,
    )
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    print("✅ Qwen2.5-7B 載入完成")
except Exception as e:
    print(f"❌ Qwen2.5-7B 載入失敗: {e}")
    sys.exit(1)

# --- 1D: Pyannote 說話者分離（cuda:0）---
print(f"⏳ [4/4] 載入 Pyannote 3.1 on {DEVICE_ASR}...")
pyannote_pipeline = None
try:
    pyannote_pipeline = PyannotePipeline.from_pretrained(
        "/data/ai_models/pyannote-speaker-diarization-3.1/config.yaml"
    )
    pyannote_pipeline = pyannote_pipeline.to(torch.device(DEVICE_ASR))
    print("✅ Pyannote 3.1 載入完成")
except Exception as e:
    print(f"⚠️  Pyannote 載入失敗，將跳過說話者分離: {e}")
    print("   → 請確認已下載模型並設定正確路徑")

print("\n🎉 所有模型載入完畢，系統就緒！\n")

# ==========================================
# 2. 工具函式
# ==========================================

def preprocess_audio(audio_path: str, target_sr: int = 16000) -> str:
    """
    音訊預處理：
    - 重採樣至 16kHz（ASR 模型標準輸入）
    - 轉為單聲道
    - 正規化音量
    輸出暫存檔路徑
    """
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    # Peak normalization
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) * 0.95
    out_path = "/tmp/preprocessed_audio.wav"
    sf.write(out_path, y, target_sr)
    return out_path


def run_diarization(audio_path: str, num_speakers: int = None) -> list:
    """
    執行 Pyannote 說話者分離。
    回傳格式：[(start_sec, end_sec, speaker_label), ...]
    """
    if pyannote_pipeline is None:
        return []
    try:
        params = {}
        if num_speakers:
            params["num_speakers"] = num_speakers
        torch.cuda.empty_cache()
        diarization = pyannote_pipeline(audio_path, **params)
        torch.cuda.empty_cache()
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # 過濾掉太短的片段（小於0.5秒）
            if turn.end - turn.start < 0.5:
                continue
            # 合併同一說話者的連續片段（間隔小於1秒）
            if segments and segments[-1][2] == speaker and turn.start - segments[-1][1] < 1.0:
                segments[-1] = (segments[-1][0], turn.end, speaker)
            else:
                segments.append((turn.start, turn.end, speaker))
        return segments
    except Exception as e:
        print(f"⚠️  Diarization 失敗: {e}")
        return []


def transcribe_segment_funasr(audio_path: str, start: float, end: float, lang_code: str) -> tuple:
    """使用 FunASR 轉寫單一音訊片段，回傳 (text, confidence)"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True,
                             offset=start, duration=end - start)
        if len(y) < 800:
            return "", 0.0
        temp_path = "/tmp/segment_temp.wav"
        sf.write(temp_path, y, sr)

        # 轉寫前清理 cuda:0 碎片化記憶體
        torch.cuda.empty_cache()

        res = sense_voice_model.generate(
            input=temp_path, cache={}, language=lang_code,
            use_itn=True, batch_size_s=16   # 降低 batch_size_s 減少顯存峰值
        )
        raw = res[0]["text"]
        text = re.sub(r'<\|.*?\|>', '', raw).strip()
        conf = min(95.0, 75.0 + len(text) * 0.1) if text else 0.0
        return text, conf
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"⚠️  VRAM 不足，嘗試縮短片段重試...")
        # OOM 時 fallback：只取片段前半段
        try:
            mid = start + (end - start) / 2
            y, sr = librosa.load(audio_path, sr=16000, mono=True,
                                 offset=start, duration=mid - start)
            sf.write(temp_path, y, sr)
            torch.cuda.empty_cache()
            res = sense_voice_model.generate(input=temp_path, cache={}, language=lang_code, use_itn=True)
            raw = res[0]["text"]
            text = re.sub(r'<\|.*?\|>', '', raw).strip()
            return text + "...[片段截斷]", 50.0
        except Exception:
            return "[顯存不足，略過此段]", 0.0
    except Exception as e:
        return f"[轉寫錯誤: {e}]", 0.0


def transcribe_segment_whisper(audio_path: str, start: float, end: float,
                                lang: str, use_tw_model: bool = False) -> tuple:
    """使用 Whisper 轉寫單一音訊片段，回傳 (text, confidence)"""
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True,
                             offset=start, duration=end - start)
        if len(y) < 800:
            return "", 0.0
        temp_path = "/tmp/segment_temp_w.wav"
        sf.write(temp_path, y, sr)
        pipe = whisper_tw_pipe if (use_tw_model and whisper_tw_pipe) else whisper_pipe
        res = pipe(
            temp_path,
            chunk_length_s=30,
            generate_kwargs={"language": lang, "task": "transcribe"},
            return_timestamps=True,
        )
        text = res["text"].strip()
        # 從 chunk logprob 計算真實信心分數
        chunks = res.get("chunks", [])
        probs = [c.get("avg_logprob", -1.0) for c in chunks if "avg_logprob" in c]
        if probs:
            avg = sum(probs) / len(probs)
            conf = round(math.exp(avg) * 100, 1)
        else:
            conf = 75.0
        return text, conf
    except Exception as e:
        return f"[轉寫錯誤: {e}]", 0.0


def format_timestamp(seconds: float) -> str:
    """將秒數格式化為 [MM:SS]"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"[{m:02d}:{s:02d}]"


def assign_speaker_labels(speakers: list) -> dict:
    """
    將 Pyannote 的 SPEAKER_00, SPEAKER_01... 映射為 A方, B方, C方...
    """
    label_map = {}
    for idx,spk in enumerate(speakers):
        if spk not in label_map:
            label_map[spk] = f"說話者{idx+1}"
    return label_map


# ==========================================
# 3. LLM 兩階段處理
# ==========================================

BATCH_CHARS = 600 # 每批最多字元數（約 2~3 分鐘對話）
def split_transcript_into_batches(transcript: str, batch_chars: int = BATCH_CHARS) -> list:
    """
    將逐字稿按行切分成多個批次，每批不超過 batch_chars 字元。
    盡量在說話者換行處切割，保持對話完整性。
    """
    lines = transcript.strip().split("\n")
    batches = []
    current = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > batch_chars and current:
            batches.append("\n".join(current))
            current = [line]
            current_len = len(line)
        else:
            current.append(line)
            current_len += len(line)
            
    if current:
        batches.append("\n".join(current))
    
    return batches

def llm_correct_transcript(raw_transcript: str, dialect: str) -> str:
    """
    第一階段：文字校正
    將方言 ASR 輸出修正為標準繁體中文逐字稿，保留對話結構。
    """
    import re
    # 拆解每行：分離「前綴」和「說話內容」
    lines = raw_transcript.strip().split("\n")
    parsed = []
    for line in lines:
        # 匹配 [時間]說話者X： 格式
        m = re.match(r'^(\[.*?\][^\：]*[：:]\s*)(.*)', line)
        if m:
            parsed.append({"prefix": m.group(1), "text": m.group(2)})
        else:
            parsed.append({"prefix": "", "text": line})
    # 只取文字內容，批次送給 LLM 校正
    system_prompt = f"""你是一位專業的中文語言學家，專精於{dialect}方言轉寫校正。
    【任務】
    以下是語音辨識產生的文字內容列表，每行是一句話。
    請將每行校正為標準「台灣繁體中文」。
    【規則】
    1. 輸入幾行就輸出幾行，行數必須完全一致。
    2. 只修正錯別字、同音字、方言用字。
    3. 簡體字全部轉為繁體字。
    4. 不要加入任何說明、編號或額外文字。
    5. 直接逐行輸出校正後的文字。"""
    # 分批處理
    BATCH_SIZE = 15 # 每批15行
    corrected_texts = []
    batches = [parsed[i:i+BATCH_SIZE] for i in range(0, len(parsed), BATCH_SIZE)]
    print(f" → 校正分批：共 {len(batches)} 批，每批 {BATCH_SIZE} 行")
    for i, batch in enumerate(batches):
        print(f" → 校正第 {i+1}/{len(batches)} 批...")
        text_lines = [item["text"] for item in batch]
        input_text = "\n".join(text_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        result = _run_llm(messages, max_tokens=600, temperature=0.1)
        # 拆回各行
        result_lines = result.strip().split("\n")
        
        # 若 LLM 輸出行數不符，直接用原始文字補齊
        if len(result_lines) != len(batch):
            print(f" ⚠️ 第{i+1}批行數不符（輸入{len(batch)}行，輸出{len(result_lines)}行），使用原始文字")
            result_lines = text_lines
        corrected_texts.extend(result_lines)
    # 拼回前綴
    final_lines = []
    for item, corrected in zip(parsed, corrected_texts):
        final_lines.append(f"{item['prefix']}{corrected}")
        
    return "\n".join(final_lines)
    
def llm_analyze_content(corrected_transcript: str, analysis_mode: str, dialect: str) -> str:
    """
    第二階段：內容分析
    基於已校正的乾淨逐字稿進行摘要與分析。
    """
    mode_instructions = {
        "重點摘要": "請針對對話內容，條列式整理主要討論事項與結論。",
        "說話意圖分析": "請分析每位說話者的溝通意圖、情緒狀態與潛在立場。",
        "全文翻譯（方言→繁中）": f"原文為{dialect}方言轉寫，請將語意翻譯為流暢的台灣繁體中文書面語，保留對話結構。",
        "情境研判報告": "請綜合分析對話背景、核心議題、各方立場，並給出情境評估。",
    }

    instruction = mode_instructions.get(analysis_mode, mode_instructions["重點摘要"])

    # 分析用的逐字稿限制在 1200 字以內，取頭尾各半
    if len(corrected_transcript) > 1200:
        half = 600
        analysis_input = (
            corrected_transcript[:half]
            + "\n...(中段省略)...\n"
            + corrected_transcript[-half:]
            )
    else:
        analysis_input = corrected_transcript

    system_prompt = f"""你是一位專業的中文對話分析師，熟悉台灣繁體中文語境。
    【任務】{instruction}

    【輸出規範】
    - 使用繁體中文。
    - 結構清晰，分段呈現。
    - 不引用逐字稿原文，用自己的語言描述。
    - 禁止使用任何 Markdown 符號，包括 #、*、**、- 等，純文字輸出。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"對話逐字稿：\n\n{analysis_input}"}
    ]
    return _run_llm(messages, max_tokens=1000, temperature=0.3)
    
def _run_llm(messages: list, max_tokens: int = 1000, temperature: float = 0.3) -> str:
    """LLM 推理共用函式"""
    try:
        torch.cuda.empty_cache()
        text_input = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = llm_tokenizer(
            [text_input], return_tensors="pt",
            truncation=True, max_length=1536
            )
        model_inputs = inputs.to(DEVICE_LLM)
        attention_mask = model_inputs["input_ids"].ne(llm_tokenizer.pad_token_id).long()
        
        with torch.no_grad():
            generated_ids = llm_model.generate(
                model_inputs.input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=llm_tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=6,
            )
        new_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        return llm_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "[LLM OOM：逐字稿過長，請縮短音訊後重試]"
    except Exception as e:
        return f"[LLM 處理錯誤: {e}]"

# ==========================================
# 4. 主處理流程
# ==========================================

# 語言路由設定表
LANG_CONFIG = {
    # 名稱 → (asr_engine, asr_lang_code, use_tw_whisper, display_name)
    "自動偵測（混合語言）":   ("funasr",  "auto", False, "混合語言"),
    "台語 / 台灣閩南語":      ("funasr",  "zh",   False, "台語"),
    "台灣國語":               ("funasr",  "zh",   False, "台灣國語"),
    "標準普通話":             ("funasr",  "zh",   False, "普通話"),
    "廣東話（粵語）":         ("funasr",  "yue",  False, "粵語"),
    "四川話":                 ("funasr",  "zh",   False, "四川話"),
    "上海話（吳語）":         ("funasr",  "zh",   False, "吳語"),
    "山東話":                 ("funasr",  "zh",   False, "山東話"),
    "中英混合":               ("funasr",  "auto", False, "中英混合"),
    "英語":                   ("whisper", "english",  False, "英語"),
    "日語":                   ("whisper", "japanese", False, "日語"),
    "韓語":                   ("whisper", "korean",   False, "韓語"),
    "俄語":                   ("whisper", "russian",  False, "俄語"),
    "藏語":                   ("whisper", "tibetan",  False, "藏語"),
    "維吾爾語":               ("whisper", "uyghur",   False, "維吾爾語"),
}


def process_audio(audio_path, target_lang, analysis_mode, num_speakers_str):
    """主處理函式（Gradio callback）"""

    if audio_path is None:
        return "<p>請上傳音訊檔案</p>", "", "", "", ""

    # 解析說話者數量
    num_speakers = None
    if num_speakers_str and num_speakers_str != "自動偵測":
        try:
            num_speakers = int(num_speakers_str)
        except ValueError:
            pass

    config = LANG_CONFIG.get(target_lang, LANG_CONFIG["標準普通話"])
    asr_engine, lang_code, use_tw, dialect_name = config

    print(f"\n{'='*50}")
    print(f"🎤 任務開始 | 語言: {target_lang} | 引擎: {asr_engine} | 模式: {analysis_mode}")

    # --- 步驟 1：音訊預處理 ---
    print("📐 步驟 1/4：音訊預處理...")
    try:
        clean_audio = preprocess_audio(audio_path)
    except Exception as e:
        clean_audio = audio_path
        print(f"⚠️  預處理失敗，使用原始音訊: {e}")

    # --- 步驟 2：說話者分離 ---
    print("👥 步驟 2/4：說話者分離（Pyannote）...")
    segments = run_diarization(clean_audio, num_speakers=num_speakers)

    if not segments:
        print("⚠️  無分離結果，改用整段轉寫模式")
        segments = [(0.0, librosa.get_duration(path=clean_audio), "SPEAKER_00")]

    # --- 步驟 3：逐段 ASR 轉寫 ---
    print(f"🎙️  步驟 3/4：逐段 ASR 轉寫（{asr_engine}，共 {len(segments)} 段）...")

    all_speakers = list(set(spk for _, _, spk in segments))
    speaker_labels = assign_speaker_labels(all_speakers)

    raw_lines = []
    confidence_scores = []

    for start, end, spk in segments:
        ts = format_timestamp(start)
        label = speaker_labels.get(spk, spk)

        if asr_engine == "funasr":
            text, conf = transcribe_segment_funasr(clean_audio, start, end, lang_code)
        else:
            text, conf = transcribe_segment_whisper(clean_audio, start, end, lang_code, use_tw)

        if text:
            raw_lines.append(f"{ts}{label}：{text}")
            confidence_scores.append(conf)

    raw_transcript = "\n".join(raw_lines) if raw_lines else "（無有效轉寫結果）"

    # 計算平均信心分數
    avg_conf = round(sum(confidence_scores) / len(confidence_scores), 1) if confidence_scores else 0.0

    # --- 步驟 4：LLM 兩階段處理 ---
    print("🧠 步驟 4/4：LLM 校正與分析...")
    print("  → 第一階段：文字校正...")
    corrected_transcript = llm_correct_transcript(raw_transcript, dialect_name)
    print("  → 第二階段：內容分析...")
    analysis_result = llm_analyze_content(corrected_transcript, analysis_mode, dialect_name)

    print("✅ 任務完成！")

    # --- 信心度儀表板 HTML ---
    if avg_conf >= 80:
        conf_color = "#16a34a"
        conf_label = "高"
    elif avg_conf >= 60:
        conf_color = "#d97706"
        conf_label = "中"
    else:
        conf_color = "#dc2626"
        conf_label = "低"

    speaker_count = len(speaker_labels)
    seg_count = len(segments)
    pyannote_status = "✅ 啟用" if pyannote_pipeline else "⚠️ 未啟用（單說話者模式）"

    stats_html = f"""
    <div style='padding:14px; border-radius:12px; background:#f8fafc; border:1px solid #e2e8f0; font-family:sans-serif;'>
        <div style='display:flex; gap:16px; align-items:center; flex-wrap:wrap;'>

            <div style='text-align:center; padding:10px 18px; background:white; border-radius:10px;
                        border:2px solid {conf_color}; min-width:90px;'>
                <div style='font-size:11px; color:#888;'>ASR 信心度</div>
                <div style='font-size:32px; font-weight:bold; color:{conf_color};'>{avg_conf}%</div>
                <div style='font-size:11px; color:{conf_color};'>{conf_label}</div>
            </div>

            <div style='text-align:center; padding:10px 18px; background:white; border-radius:10px;
                        border:2px solid #3b82f6; min-width:90px;'>
                <div style='font-size:11px; color:#888;'>說話者</div>
                <div style='font-size:32px; font-weight:bold; color:#3b82f6;'>{speaker_count}</div>
                <div style='font-size:11px; color:#3b82f6;'>位</div>
            </div>

            <div style='text-align:center; padding:10px 18px; background:white; border-radius:10px;
                        border:2px solid #8b5cf6; min-width:90px;'>
                <div style='font-size:11px; color:#888;'>語音段數</div>
                <div style='font-size:32px; font-weight:bold; color:#8b5cf6;'>{seg_count}</div>
                <div style='font-size:11px; color:#8b5cf6;'>段</div>
            </div>

            <div style='flex:1; min-width:160px; font-size:12px; color:#555; line-height:1.8;'>
                <div>🔊 ASR 引擎：<b>{asr_engine.upper()}</b></div>
                <div>🧠 LLM：<b>Qwen2.5-7B</b></div>
                <div>👥 說話者分離：<b>{pyannote_status}</b></div>
                <div>🌐 語言：<b>{target_lang}</b> ｜ 模式：<b>{analysis_mode}</b></div>
            </div>
        </div>
    </div>
    """

    return stats_html, raw_transcript, corrected_transcript, analysis_result, corrected_transcript


# ==========================================
# 5. AI 對話修正函式
# ==========================================

def chat_with_ai(user_message, chat_history, current_transcript):
    """
    AI 對話修正框：讓使用者針對輸出結果進行追問或修正指令。
    current_transcript 是當前校正後的逐字稿，作為對話上下文。
    """
    if not user_message.strip():
        return chat_history, ""

    system_prompt = """你是一位專業的中文逐字稿校對助理。
使用者會針對語音轉寫的結果提出修正要求或問題。
請根據上下文中的逐字稿內容，依照使用者的指示進行修改或回答。
回覆使用繁體中文，保持簡潔清楚。"""

    context = f"【當前逐字稿內容】\n{current_transcript}\n\n" if current_transcript else ""

    # 組建對話歷史給 LLM
    messages = [{"role": "system", "content": system_prompt}]
    for human, assistant in chat_history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": context + user_message})

    response = _run_llm(messages, max_tokens=1000, temperature=0.4)

    chat_history.append((user_message, response))
    return chat_history, ""


# ==========================================
# 6. Gradio 介面
# ==========================================
with gr.Blocks(title="雅言 AI V4.1", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🏺 雅言 AI — 方言對話轉寫系統 V4.1")
    gr.Markdown(
        "**FunASR SenseVoice + Whisper V3 + Pyannote 說話者分離 + Qwen2.5-7B 校正**"
    )

    # 隱藏狀態：儲存校正後逐字稿供對話框使用
    current_corrected = gr.State("")

    with gr.Row():
        # --- 左側控制欄 ---
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="🎵 上傳音訊 / 錄音"
            )
            lang_dropdown = gr.Dropdown(
                choices=list(LANG_CONFIG.keys()),
                value="台語 / 台灣閩南語",
                label="🌐 來源語言"
            )
            num_speakers_dropdown = gr.Dropdown(
                choices=["自動偵測", "2", "3", "4", "5"],
                value="自動偵測",
                label="👥 說話者人數"
            )
            mode_dropdown = gr.Dropdown(
                choices=["重點摘要", "說話意圖分析", "全文翻譯（方言→繁中）", "情境研判報告"],
                value="重點摘要",
                label="🧠 分析模式"
            )
            submit_btn = gr.Button("🚀 開始轉寫分析", variant="primary", size="lg")

            # 信心度儀表板（轉寫後顯示）
            stats_out = gr.HTML(label="📊 分析統計")

        # --- 右側輸出欄 ---
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📝 原始逐字稿"):
                    raw_out = gr.Textbox(
                        label="ASR 輸出（含時間戳記與說話方）",
                        lines=15,
                        interactive=False,
                        placeholder="[00:00]A方：...\n[00:05]B方：..."
                    )
                with gr.Tab("✅ 校正後逐字稿"):
                    corrected_out = gr.Textbox(
                        label="Qwen2.5-7B 方言校正結果",
                        lines=15,
                        interactive=True,   # 允許手動微調
                    )
                with gr.Tab("📊 分析報告"):
                    analysis_out = gr.Textbox(
                        label="內容分析結果",
                        lines=15,
                        interactive=False
                    )

    # --- AI 對話修正框 ---
    gr.Markdown("---\n### 💬 AI 修正對話框")
    gr.Markdown("針對轉寫結果進行追問或修正，AI 會參考當前校正後的逐字稿內容回覆。")

    chatbot = gr.Chatbot(label="對話記錄", height=280)

    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="例：幫我把第3行的「你好」改成「妳好」/ 這段對話的主題是什麼？",
            label="輸入修正指令或問題",
            scale=4,
            lines=2,
        )
        chat_btn = gr.Button("送出", variant="primary", scale=1, min_width=80)
        clear_btn = gr.Button("清除對話", scale=1, min_width=80)

    # --- 事件綁定 ---
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, lang_dropdown, mode_dropdown, num_speakers_dropdown],
        outputs=[stats_out, raw_out, corrected_out, analysis_out, current_corrected],
    )

    chat_btn.click(
        fn=chat_with_ai,
        inputs=[chat_input, chatbot, current_corrected],
        outputs=[chatbot, chat_input],
    )

    chat_input.submit(
        fn=chat_with_ai,
        inputs=[chat_input, chatbot, current_corrected],
        outputs=[chatbot, chat_input],
    )

    clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, chat_input])

    gr.Markdown("""
    ---
    **輸出格式：** `[00:00]A方：你好` → `[00:03]B方：你好啊`　｜　校正後逐字稿可直接在文字框內手動編輯。
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
