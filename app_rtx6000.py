"""
YaYan-AI v4.6 — Quadro RTX 6000 (Turing) x 2 主介面
"""
from __future__ import annotations

import os
import sys
import json
import logging
import statistics
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_DO_NOT_TRACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

import gradio as gr

from yayan import __version__
from yayan.config import CONFIG
from yayan.pipeline import (
    transcribe_audio,
    refine_with_user_edit,
    warmup,
    TranscriptionResult,
)


# ★ v4.6: 22 中文方言全部展開
DIALECT_TO_ROUTING = {
    "🔍 自動偵測（建議：逐段 LID）": "auto",
    "🇹🇼 國台混合 / 多方言混雜": "auto",
    # ── 北方官話 ──
    "🇨🇳 北京話 / 普通話": "zh",
    "🇨🇳 東北話": "cmn-ne",
    "🇨🇳 山東話": "cmn-sd",
    "🇨🇳 河南話": "cmn-zy",
    "🇨🇳 西安話": "cmn-xa",
    "🇨🇳 蘭州話": "cmn-lz",
    # ── 西南官話 ──
    "🇨🇳 四川話 (西南官話)": "cmn-sw",
    "🇨🇳 武漢話": "cmn-wh",
    # ── 江淮官話 ──
    "🇨🇳 南京話": "cmn-jh",
    # ── 吳語 ──
    "🇨🇳 上海話 (吳語)": "wuu",
    "🇨🇳 蘇州話": "wuu-sz",
    "🇨🇳 寧波話": "wuu-nb",
    "🇨🇳 溫州話": "wuu-wz",
    # ── 粵語 ──
    "🇭🇰 廣東話 (粵語)": "yue",
    # ── 閩語 ──
    "🇹🇼 閩南語 / 台語": "nan",
    "🇨🇳 潮汕話 / 潮州話": "nan-cs",
    "🇨🇳 海南話": "nan-hn",
    "🇨🇳 福州話 (閩東語)": "cdo",
    # ── 客贛湘晉 ──
    "🇨🇳 客家話": "hak",
    "🇨🇳 湖南話 (湘語)": "hsn",
    "🇨🇳 江西話 (贛語)": "gan",
    "🇨🇳 山西話 (晉語)": "cjy",
    # ── 中國少數民族 ──
    "🏔️ 藏語 (Tibetan)": "bo",
    "🌙 維吾爾語 (Uyghur)": "ug",
    # ── 東亞 ──
    "🇯🇵 日文 (Japanese)": "ja",
    "🇰🇷 韓文 (Korean)": "ko",
    # ── 中東 / 南亞 ──
    "🇮🇷 波斯語 (Farsi)": "fa",
    "🇵🇰 烏爾都語 (Urdu)": "ur",
    "🇸🇦 阿拉伯語 (Arabic)": "ar",
    "🇮🇳 印地語 (Hindi)": "hi",
    # ── 歐洲 ──
    "🇬🇧 英語 (English)": "en",
    "🇫🇷 法語 (French)": "fr",
    "🇩🇪 德語 (German)": "de",
    "🇷🇺 俄語 (Russian)": "ru",
    "🇪🇸 西班牙語 (Spanish)": "es",
    # ── 東南亞 ──
    "🇹🇭 泰語 (Thai)": "th",
    "🇲🇾 馬來語 (Malay)": "ms",
    "🇻🇳 越南語 (Vietnamese)": "vi",
    "🇮🇩 印尼語 (Indonesian)": "id",
}
DIALECT_CHOICES = list(DIALECT_TO_ROUTING.keys())

ROUTING_DISPLAY = {
    "zh": "普通話", "cmn": "普通話",
    "cmn-ne": "東北話", "cmn-sd": "山東話", "cmn-zy": "河南話",
    "cmn-xa": "西安話", "cmn-lz": "蘭州話",
    "cmn-sw": "四川話", "cmn-wh": "武漢話", "cmn-jh": "南京話",
    "yue": "粵語",
    "wuu": "上海話", "wuu-sz": "蘇州話", "wuu-nb": "寧波話", "wuu-wz": "溫州話",
    "nan": "閩南語/台語", "nan-cs": "潮汕話", "nan-hn": "海南話",
    "cdo": "福州話",
    "hak": "客家話", "hsn": "湘語", "gan": "贛語", "cjy": "晉語",
    "bo": "藏語", "ug": "維吾爾語",
    "ja": "日文", "ko": "韓文",
    "fa": "波斯語", "ur": "烏爾都語", "ar": "阿拉伯語", "hi": "印地語",
    "en": "英語", "fr": "法語", "de": "德語", "ru": "俄語", "es": "西班牙語",
    "th": "泰語", "ms": "馬來語", "vi": "越南語", "id": "印尼語",
    "auto": "未識別",
}


def _calc_confidence(result: TranscriptionResult) -> tuple[float, str]:
    segments = result.segments
    raw_text = result.raw_text or ""
    translated = result.translated_text or ""
    if not segments:
        return 0.0, "無段落"

    notes = []
    score = 100.0
    durations = [s.end - s.start for s in segments]
    total_dur = sum(durations)

    if total_dur > 0:
        seg_per_min = len(segments) / (total_dur / 60)
        if seg_per_min > 80:
            score -= 15; notes.append("VAD 切片過密")
        elif seg_per_min < 5:
            score -= 10; notes.append("VAD 切片過疏")

    avg_dur = statistics.mean(durations)
    if avg_dur < 0.8:
        score -= 15; notes.append("段落過短")
    elif avg_dur > 20:
        score -= 10; notes.append("段落過長")

    if len(durations) > 1:
        std = statistics.stdev(durations)
        cv = std / avg_dur if avg_dur > 0 else 0
        if cv > 1.5:
            score -= 10; notes.append("段長不穩")

    seg_texts = [s.raw_text.strip() for s in segments if s.raw_text.strip()]
    if seg_texts:
        unique_ratio = len(set(seg_texts)) / len(seg_texts)
        if unique_ratio < 0.6:
            score -= 20; notes.append("ASR 重複嚴重")
        elif unique_ratio < 0.8:
            score -= 10; notes.append("ASR 些許重複")

    if raw_text and translated:
        ratio = len(translated) / max(len(raw_text), 1)
        if ratio < 0.5:
            score -= 15; notes.append("譯文偏短")
        elif ratio > 2.0:
            score -= 10; notes.append("譯文偏長")
    elif raw_text and not translated:
        score -= 30; notes.append("LLM 翻譯失敗")

    lid_confs = [s.lid_conf for s in segments if s.lid_conf > 0]
    if lid_confs:
        avg_lid = statistics.mean(lid_confs)
        if avg_lid < 0.4:
            score -= 15; notes.append(f"LID 信心低({avg_lid:.2f})")
        elif avg_lid < 0.5:
            score -= 5; notes.append(f"LID 信心中({avg_lid:.2f})")
    
    fallback_count = sum(1 for s in segments if s.lid_method in ("fallback", "lid_error"))
    if segments and fallback_count / len(segments) > 0.3:
        score -= 10
        notes.append(f"LID 失敗段過多")

    score = max(0.0, min(100.0, score))
    note_text = "、".join(notes) if notes else "品質良好"
    return round(score, 1), note_text


def _format_lang_breakdown(breakdown: dict) -> str:
    if not breakdown:
        return ""
    parts = []
    total = sum(breakdown.values())
    for code, count in sorted(breakdown.items(), key=lambda x: -x[1])[:5]:
        name = ROUTING_DISPLAY.get(code, code)
        pct = count / total * 100
        parts.append(f"{name} {count}({pct:.0f}%)")
    return "｜".join(parts)


def fn_transcribe(audio_path, dialect_label, enable_diarize):
    if audio_path is None:
        return "請先上傳或錄製音檔。", "", "", "", "—"

    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        result: TranscriptionResult = transcribe_audio(
            audio_path,
            language=routing,
            use_diarize=enable_diarize,
        )
    except Exception as e:
        logging.exception("transcribe 失敗")
        return f"識別失敗：{e}", "", "", "", "—"

    breakdown_text = _format_lang_breakdown(result.language_breakdown)
    n_speakers = len(set(s.speaker for s in result.segments)) if result.segments else 0
    info = (
        f"📊 語言分布：{breakdown_text}\n"
        f"👥 說話人：{n_speakers} 位｜段數：{len(result.segments)}"
    )
    confidence, note = _calc_confidence(result)
    score_text = f"{confidence:.1f} / 100\n{note}"
    return info, result.raw_text, result.raw_text, result.translated_text, score_text


def fn_refine(raw_text_original, edited_text, dialect_label):
    if not edited_text or not edited_text.strip():
        return gr.update()
    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        refined = refine_with_user_edit(
            raw_text=raw_text_original or edited_text,
            user_edit=edited_text,
            source_language=routing,
        )
    except Exception as e:
        logging.exception("refine 失敗")
        return f"重新潤飾失敗：{e}"
    return refined


def fn_save_as(translated_text, source_audio):
    if not translated_text or not translated_text.strip():
        return None
    out_dir = Path(CONFIG["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(source_audio).stem if source_audio else "manual"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"{base}_{ts}.txt"
    out.write_text(translated_text, encoding="utf-8")
    return str(out)


CSS = """
.yayan-title { font-size: 1.5em; font-weight: 600; }
.yayan-sub   { color: #888; font-size: 0.9em; }
.confidence-box textarea {
    text-align: center !important;
    font-size: 1.4em !important;
    font-weight: 700 !important;
    color: #2563eb !important;
    line-height: 1.3 !important;
}
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"YaYan-AI v{__version__}", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🏺 YaYan-AI **v{__version__}**　— 多語言情報系統
            <p class="yayan-sub">Edition: RTX6000-Server　|　ASR: Dolphin-CN-Dialect (22 方言) / Whisper-large-v3　|　LLM: Qwen3-14B + NF4　|　逐段 LID + 5 人說話人分離 + 字級時間戳</p>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="🎤 上傳或錄製音檔",
                )
                dialect = gr.Dropdown(
                    choices=DIALECT_CHOICES,
                    value="🔍 自動偵測（建議：逐段 LID）",
                    label="來源語言",
                )
                enable_diarize = gr.Checkbox(
                    label="啟用說話人分離（A方/B方/C方/D方/E方）",
                    value=True,
                )
                transcribe_btn = gr.Button("🚀 開始轉錄翻譯", variant="primary", size="lg")
                info_box = gr.Textbox(label="識別資訊", interactive=False, lines=3)

                confidence_box = gr.Textbox(
                    label="🎯 識別精準度",
                    value="—",
                    interactive=False,
                    lines=2,
                    elem_classes=["confidence-box"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### 📜 識別原文（可編輯，含時間戳）")
                raw_text_display = gr.State("")
                raw_text_box = gr.Textbox(
                    label="ASR 原文",
                    lines=10,
                    interactive=True,
                    placeholder="格式：[A方 00:01-00:05] 你好",
                )

                gr.Markdown("### 🇹🇼 台灣正體中文譯文（可編輯）")
                translated_box = gr.Textbox(
                    label="譯文",
                    lines=10,
                    interactive=True,
                    placeholder="翻譯結果會保留 [A方 00:01-00:05] 標籤",
                )

                with gr.Row():
                    refine_raw_btn = gr.Button("🔄 依【編輯後原文】重新翻譯潤飾", variant="secondary")
                    refine_translated_btn = gr.Button("✨ 依【編輯後譯文】重新潤飾", variant="secondary")

                save_btn = gr.Button("💾 另存新檔", variant="primary")
                save_file = gr.File(
                    label="📁 點擊下方檔案連結即可選擇儲存位置",
                    interactive=False,
                )

        transcribe_btn.click(
            fn=fn_transcribe,
            inputs=[audio_input, dialect, enable_diarize],
            outputs=[info_box, raw_text_display, raw_text_box, translated_box, confidence_box],
        )
        refine_raw_btn.click(
            fn=fn_refine,
            inputs=[raw_text_display, raw_text_box, dialect],
            outputs=[translated_box],
        )
        refine_translated_btn.click(
            fn=fn_refine,
            inputs=[raw_text_display, translated_box, dialect],
            outputs=[translated_box],
        )
        save_btn.click(
            fn=fn_save_as,
            inputs=[translated_box, audio_input],
            outputs=[save_file],
        )

        gr.Markdown(
            """
            ---
            **v4.6 新功能：**
            - **22 種中文方言**：含福州話、客家話、湘贛晉、潮汕、海南、蘇杭吳語細分等
            - **字級時間戳**：每段顯示 `[A方 00:01-00:05] 內容`
            - **逐段語言識別**：混合語音每段獨立判斷
            - **5 人說話人分離**：A方 / B方 / C方 / D方 / E方
            - **語言分布統計**：左上顯示音檔內各語言比例

            **ASR 引擎：**
            - 漢語方言（22 種） + 藏維 → **Dolphin-CN-Dialect-Small**（清華 + DataoceanAI）
            - 日韓 + 歐洲 + 中東 + 東南亞 → **Whisper-large-v3**
            """
        )
    return demo


def main():
    server_cfg = CONFIG["server"]
    print("=" * 60)
    print(f"  YaYan-AI v{__version__}  |  Edition: RTX6000-Server")
    print(f"  Models root: {CONFIG['paths']['models_root']}")
    print(f"  ASR GPU: {CONFIG['devices']['asr_gpu']}  |  LLM GPU: {CONFIG['devices']['llm_gpu']}")
    print(f"  LLM backend: {CONFIG['llm'].get('backend')}  |  quant: {CONFIG['llm'].get('quantization')}")
    print(f"  支援語言/方言數: {len(DIALECT_TO_ROUTING)}")
    print("=" * 60)

    print("⏳ 預載模型 …")
    try:
        warmup()
    except Exception as e:
        print(f"⚠️ Warmup 失敗（仍可啟動）：{e}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=server_cfg["host"],
        server_port=server_cfg["port"],
        share=server_cfg["share"],
    )


if __name__ == "__main__":
    main()
