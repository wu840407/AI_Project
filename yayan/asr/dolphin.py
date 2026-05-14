"""YaYan_ASR_Dialect — Dolphin-CN-Dialect 包裝。"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Dolphin")

_MODEL = None

ROUTING_TO_DOLPHIN: dict = {
    "zh":    ("zh", "CN"),
    "cmn":   ("zh", "CN"),
    "yue":   ("yue", "CN"),
    "wuu":   ("wuu", "CN"),
    "wuu-sz":("wuu", "CN-SZ"),
    "wuu-nb":("wuu", "CN-NB"),
    "wuu-wz":("wuu", "CN-WZ"),
    "nan":   ("nan", "CN"),
    "nan-cs":("nan", "CN-CS"),
    "nan-hn":("nan", "CN-HN"),
    "cdo":   ("cdo", "CN"),
    "hak":   ("hak", "CN"),
    "hsn":   ("hsn", "CN"),
    "gan":   ("gan", "CN"),
    "cjy":   ("cjy", "CN"),
    "cmn-sw":("zh", "CN-SC"),
    "cmn-sd":("zh", "CN-SD"),
    "cmn-ne":("zh", "CN-NE"),
    "cmn-zy":("zh", "CN-ZY"),
    "cmn-wh":("zh", "CN-WH"),
    "cmn-xa":("zh", "CN-XA"),
    "cmn-lz":("zh", "CN-LZ"),
    "cmn-jh":("zh", "CN-JH"),
    "bo":    ("bo", "CN"),
    "ug":    ("ug", "CN"),
    "min":   ("nan", "CN"),
    "hokkien":("nan", "CN"),
}


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class DolphinResult:
    text: str
    words: List[WordTimestamp]
    detected_lang: str = ""
    detected_region: str = ""


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import dolphin
    except ImportError as e:
        raise ImportError(
            "dolphin SDK 未安裝。請：pip install dataoceanai-dolphin"
        ) from e

    local_dir = model_path("YaYan_ASR_Dialect")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Dialect 不存在: {local_dir}")

    device = CONFIG["devices"]["asr_gpu"]

    pt_files = list(local_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"{local_dir} 找不到 .pt 模型權重")
    pt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    model_name = pt_files[0].stem
    logger.info(
        f"載入 YaYan_ASR_Dialect (model_name={model_name}, file={pt_files[0].name}) on {device}"
    )

    _MODEL = dolphin.load_model(model_name, str(local_dir), device)
    return _MODEL


def transcribe(
    audio: np.ndarray,
    language_hint: Optional[str] = None,
    enable_word_timestamp: bool = True,
    sample_rate: int = 16000,
) -> DolphinResult:
    import dolphin

    model = _load()
    tmp_path = None
    try:
        if isinstance(audio, np.ndarray):
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            audio = audio.astype(np.float32)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="yayan_dolphin_")
            os.close(tmp_fd)
            sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")
            audio_input = tmp_path
        else:
            audio_input = audio

        lang_sym, region_sym = ROUTING_TO_DOLPHIN.get(
            (language_hint or "auto").lower(), (None, None)
        )
        kwargs = {}
        if lang_sym:
            kwargs["lang_sym"] = lang_sym
        if region_sym:
            kwargs["region_sym"] = region_sym
        if enable_word_timestamp:
            kwargs["predict_time"] = True

        try:
            result = dolphin.transcribe(model, audio_input, **kwargs)
        except TypeError:
            kwargs.pop("predict_time", None)
            result = dolphin.transcribe(model, audio_input, **kwargs)

    except Exception as e:
        logger.error(f"Dolphin 推論失敗 (lang={language_hint}): {e}")
        return DolphinResult(text="", words=[])
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    text = (getattr(result, "text", "") or "").strip()
    text = _strip_special_tokens(text)

    words: List[WordTimestamp] = []
    raw_words = (
        getattr(result, "words", None)
        or getattr(result, "word_timestamps", None)
        or getattr(result, "tokens", None)
    )
    if raw_words:
        for w in raw_words:
            try:
                if isinstance(w, dict):
                    words.append(WordTimestamp(
                        word=_strip_special_tokens(str(w.get("word", w.get("text", "")))),
                        start=float(w.get("start", w.get("start_time", 0))),
                        end=float(w.get("end", w.get("end_time", 0))),
                    ))
                else:
                    words.append(WordTimestamp(
                        word=_strip_special_tokens(str(getattr(w, "word", getattr(w, "text", "")))),
                        start=float(getattr(w, "start", getattr(w, "start_time", 0))),
                        end=float(getattr(w, "end", getattr(w, "end_time", 0))),
                    ))
            except Exception:
                continue

    detected_lang = (
        getattr(result, "language", "")
        or getattr(result, "lang", "")
        or getattr(result, "lang_sym", "")
        or ""
    )
    detected_region = (
        getattr(result, "region", "")
        or getattr(result, "region_sym", "")
        or ""
    )

    return DolphinResult(
        text=text,
        words=words,
        detected_lang=str(detected_lang),
        detected_region=str(detected_region),
    )


def _strip_special_tokens(text: str) -> str:
    """移除 Dolphin 各種特殊 token。

    Dolphin 的 token 格式有四種：
      <|...|>               例如 <|zh|>, <|nospeech|>
      <CN>                  地區（大寫）
      <notimestamp>         長 token（11+ 字）
      <0.50>                時間戳數字
    """
    # <|...|>
    text = re.sub(r"<\|[^|]*\|>", "", text)
    # <Xxx> (任意字母數字)
    text = re.sub(r"<[A-Za-z][A-Za-z0-9_-]*>", "", text)
    # <0.50> 數字時間戳
    text = re.sub(r"<\d+(?:\.\d+)?>", "", text)
    return text.strip()
