"""ASR 路由器：依語言路由到正確的 YaYan_ASR_* 模型。"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import CONFIG

logger = logging.getLogger("YaYan.ASR.Router")


@dataclass
class AsrResult:
    text: str
    routing: str
    asr_alias: str
    confidence: float = 0.0


_DISPATCH = {
    "YaYan_ASR_Mandarin": ("sensevoice", {"zh", "yue", "wuu", "cmn", "cdo"}),
    "YaYan_ASR_Eastern": ("dolphin", {"bo", "ug", "wuu"}),
    "YaYan_ASR_Global": ("whisper_global", {"fa", "ur", "en", "ar"}),
}


def _alias_for(routing: str) -> str:
    routing_table = CONFIG["asr"]["routing"]
    return routing_table.get(routing, CONFIG["asr"]["default_alias"])


def transcribe(
    audio: np.ndarray,
    routing: str = "auto",
    language_hint: Optional[str] = None,
) -> AsrResult:
    alias = _alias_for(routing)
    logger.info(f"路由 → {alias} (routing={routing}, hint={language_hint})")

    if alias == "YaYan_ASR_Mandarin":
        from . import sensevoice
        text = sensevoice.transcribe(audio, language_hint or routing)
    elif alias == "YaYan_ASR_Eastern":
        from . import dolphin
        text = dolphin.transcribe(audio, language_hint or routing)
    elif alias == "YaYan_ASR_Global":
        from . import whisper_global
        text = whisper_global.transcribe(audio, language_hint or routing)
    else:
        raise ValueError(f"未知 ASR 別名: {alias}")

    return AsrResult(text=text, routing=routing, asr_alias=alias)
