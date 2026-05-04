"""
YaYan-AI v4.5 — Quadro RTX 6000 (Turing) x 2 主介面
功能：
  - 網頁上傳音檔（保留）
  - 自動方言識別 + 路由（北京/山東/上海/四川/廣東/維吾爾/藏 + 波斯/烏爾都）
  - 對話修改回環：使用者可編輯原文/譯文，按「重新潤飾」由 LLM 重跑後段
  - 輸出強制台灣正體中文（OpenCC）
  - 完全離線（local_files_only）
"""
from __future__ import annotations

import os
import sys
import json
import logging
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

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


# ---------------- UI 對應表（YaYan_* 對外名） ---------------- #
DIALECT_TO_ROUTING = {
    "自動偵測": "auto",
    "北京話 / 普通話": "zh",
    "山東話": "zh",
    "上海話 (吳語)": "wuu",
    "四川話": "zh",
    "廣東話 (粵語)": "yue",
    "新疆話 / 維吾爾語": "ug",
    "藏語": "bo",
    "波斯語 (Farsi)": "fa",
    "烏爾都語 (Urdu)": "ur",
    "英語": "en",
}
DIALECT_CHOICES = list(DIALECT_TO_ROUTING.keys())

ROUTING_TO_DIALECT = {v: k for k, v in DIALECT_TO_ROUTING.items()}


# ---------------- 業務邏輯 ---------------- #

def fn_transcribe(audio_path: str, dialect_label: str, enable_diarize: bool):
    if audio_path is None:
        return "請先上傳或錄製音檔。", "", "", "", "{}"

    routing = DIALECT_TO_ROUTING.get(dialect_label, "auto")
    try:
        result: TranscriptionResult = transcribe_audio(
            audio_path,
            language=routing,
            use_diarize=enable_diarize,
        )
    except Exception as e:
        logging.exception("transcribe 失敗")
        return f"識別失敗：{e}", "", "", "", "{}"

    detected = ROUTING_TO_DIALECT.get(result.routing, result.routing)
    info = f"偵測語言：{detected}（{result.routing}）｜段數：{len(result.segments)}"
    meta = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    return info, result.raw_text, result.raw_text, result.translated_text, meta


def fn_refine(raw_text_original: str, edited_text: str, dialect_label: str):
    if not edited_text or not edited_text.strip():
        return "編輯框是空的，無法重新潤飾。"
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


def fn_save(translated_text: str, source_audio: str) -> str:
    if not translated_text:
        return "沒有可儲存的內容。"
    out_dir = Path(CONFIG["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(source_audio).stem if source_audio else "manual"
    out = out_dir / f"{base}_translated.txt"
    out.write_text(translated_text, encoding="utf-8")
    return f"✅ 已儲存：{out}"


# ---------------- Gradio 介面 ---------------- #

CSS = """
.yayan-title { font-size: 1.5em; font-weight: 600; }
.yayan-sub   { color: #888; font-size: 0.9em; }
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=f"YaYan-AI v{__version__}", css=CSS, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"""
            # 🏺 YaYan-AI **v{__version__}**　— 多方言語音情報系統
            <p class="yayan-sub">Edition: RTX6000-Server　|　模型路由：YaYan_ASR_Mandarin / YaYan_ASR_Eastern / YaYan_ASR_Global　|　LLM：YaYan_Reasoner</p>
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
                    value="自動偵測",
                    label="來源方言（可選；建議用自動）",
                )
                enable_diarize = gr.Checkbox(
                    label="啟用說話人分離（雙人通話建議）",
                    value=False,
                )
                transcribe_btn = gr.Button("🚀 開始轉錄翻譯", variant="primary", size="lg")
                info_box = gr.Textbox(label="識別資訊", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 📜 識別原文（可編輯）")
                raw_text_display = gr.State("")
                raw_text_box = gr.Textbox(
                    label="ASR 原文",
                    lines=6,
                    interactive=True,
                    placeholder="識別結果會顯示在此處，您可直接修改後按「依編輯重新潤飾」。",
                )

                gr.Markdown("### 🇹🇼 台灣正體中文譯文（可編輯）")
                translated_box = gr.Textbox(
                    label="譯文",
                    lines=8,
                    interactive=True,
                    placeholder="翻譯結果會顯示在此處，您可直接修改後按「依編輯重新潤飾」。",
                )

                with gr.Row():
                    refine_raw_btn = gr.Button("🔄 依【編輯後原文】重新翻譯潤飾", variant="secondary")
                    refine_translated_btn = gr.Button("✨ 依【編輯後譯文】重新潤飾", variant="secondary")
                    save_btn = gr.Button("💾 儲存譯文", variant="primary")

                save_status = gr.Textbox(label="狀態", interactive=False)

        with gr.Accordion("🔍 進階：完整中繼資料（JSON）", open=False):
            meta_box = gr.Code(language="json", label="metadata")

        # ---------- 事件綁定 ---------- #
        transcribe_btn.click(
            fn=fn_transcribe,
            inputs=[audio_input, dialect, enable_diarize],
            outputs=[info_box, raw_text_display, raw_text_box, translated_box, meta_box],
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
            fn=fn_save,
            inputs=[translated_box, audio_input],
            outputs=[save_status],
        )

        gr.Markdown(
            """
            ---
            **使用提示：**
            1. 上傳電話錄音 → 點「開始轉錄翻譯」 → 取得原文與譯文。
            2. 若 ASR 有錯，**直接在「ASR 原文」框修改** → 點「依編輯後原文重新翻譯潤飾」。
            3. 若譯文要微調，**直接在「譯文」框修改** → 點「依編輯後譯文重新潤飾」。
            4. 滿意後按「儲存譯文」，輸出到設定的 output_dir。
            """
        )
    return demo


def main():
    server_cfg = CONFIG["server"]

    print("=" * 60)
    print(f"  YaYan-AI v{__version__}  |  Edition: RTX6000-Server")
    print(f"  Models root: {CONFIG['paths']['models_root']}")
    print(f"  ASR GPU: {CONFIG['devices']['asr_gpu']}  |  LLM GPU: {CONFIG['devices']['llm_gpu']}")
    print("=" * 60)

    print("⏳ 預載模型 …")
    try:
        warmup()
    except Exception as e:
        print(f"⚠️ Warmup 失敗（仍可啟動，將在首次請求時載入）：{e}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=2).launch(
        server_name=server_cfg["host"],
        server_port=server_cfg["port"],
        share=server_cfg["share"],
    )


if __name__ == "__main__":
    main()
