"""YaYan ASR 路由器：依 routing key 決定走哪個 ASR 引擎。

v4.6 變動：
  - 漢語方言（22 種）→ YaYan_ASR_Dialect (Dolphin-CN-Dialect-Small)
  - 日韓 → YaYan_ASR_Global (Whisper-large-v3，比 SenseVoice 在日韓更強)
  - 中亞 (bo/ug) → YaYan_ASR_Dialect (Dolphin 也支援)
  - 其他 → YaYan_ASR_Global
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..config import CONFIG

logger = logging.getLogger("YaYan.ASR.Router")

# 漢語方言 + Dolphin 直接支援的非中文
DOLPHIN_ROUTINGS = {
    # 漢語方言
    "zh", "cmn", "yue", "wuu", "nan", "cdo", "hak", "hsn", "gan", "cjy",
    "wuu-sz", "wuu-nb", "wuu-wz",
    "nan-cs", "nan-hn",
    "cmn-sw", "cmn-sd", "cmn-ne", "cmn-zy", "cmn-wh", "cmn-xa", "cmn-lz", "cmn-jh",
    # 中亞
    "bo", "ug",
    "min", "hokkien",  # 別名
}


@dataclass
class WordTS:
    word: str
    start: float  # 全域時間（chunk_start_in_audio + word_offset_in_chunk）
    end: float


@dataclass
class ASRResult:
    text: str
    asr_alias: str
    routing: str
    words: List[WordTS] = None  # type: ignore


def transcribe(
    audio: np.ndarray,
    routing: str = "auto",
    language_hint: Optional[str] = None,
    chunk_start_sec: float = 0.0,
) -> ASRResult:
    """依 routing key 選 ASR 引擎。

    chunk_start_sec: 用於把 word timestamp 從 chunk-local 轉成 audio-global
    """
    routing = (routing or "auto").lower()
    logger.info(f"路由 → routing={routing}")

    if routing in DOLPHIN_ROUTINGS or routing == "auto":
        # 包含 auto → Dolphin 自己會做 LID
        return _dolphin_transcribe(audio, routing, chunk_start_sec)
    else:
        # 日韓 / 歐美 / 中東 → Whisper
        return _whisper_transcribe(audio, routing, chunk_start_sec)


def _dolphin_transcribe(
    audio: np.ndarray, routing: str, chunk_start_sec: float,
) -> ASRResult:
    from . import dolphin as _dolphin_mod
    res = _dolphin_mod.transcribe(audio, language_hint=routing)
    words = [
        WordTS(
            word=w.word,
            start=w.start + chunk_start_sec,
            end=w.end + chunk_start_sec,
        )
        for w in (res.words or [])
    ]
    return ASRResult(
        text=res.text,
        asr_alias="YaYan_ASR_Dialect",
        routing=routing,
        words=words,
    )


def _whisper_transcribe(
    audio: np.ndarray, routing: str, chunk_start_sec: float,
) -> ASRResult:
    """Whisper 路徑（日韓、波斯、英、法、德、俄、泰、馬來等）。"""
    from . import whisper_global as _whisper
    res = _whisper.transcribe(audio, language_hint=routing)

    # 嘗試從 Whisper 取 word timestamp
    words = []
    if hasattr(res, "words") and res.words:
        for w in res.words:
            try:
                words.append(WordTS(
                    word=str(w["word"]) if isinstance(w, dict) else str(w.word),
                    start=float(w["start"] if isinstance(w, dict) else w.start) + chunk_start_sec,
                    end=float(w["end"] if isinstance(w, dict) else w.end) + chunk_start_sec,
                ))
            except Exception:
                continue
    text = getattr(res, "text", "") if hasattr(res, "text") else str(res)

    return ASRResult(
        text=text,
        asr_alias="YaYan_ASR_Global",
        routing=routing,
        words=words,
    )
