"""YaYan_ASR_Eastern — 東方語族 ASR（藏語、維吾爾語、吳語等）。"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Eastern")

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import dolphin

    local_dir = model_path("YaYan_ASR_Eastern")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Eastern 不存在: {local_dir}")
    logger.info("載入 YaYan_ASR_Eastern …")
    device = CONFIG["devices"]["asr_gpu"]
    _MODEL = dolphin.load_model("base", str(local_dir), device)
    return _MODEL


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    """language_hint: bo/ug/wuu/zh"""
    model = _load()
    lang_map = {"bo": "bo", "ug": "ug", "wuu": "zh", "zh": "zh"}
    lang_sym = lang_map.get((language_hint or "").lower(), None)

    try:
        if lang_sym:
            res = model.transcribe(audio, lang_sym=lang_sym)
        else:
            res = model.transcribe(audio)
    except TypeError:
        res = model(audio)

    if hasattr(res, "text"):
        return res.text.strip()
    if isinstance(res, dict):
        return res.get("text", "").strip()
    return str(res).strip()
