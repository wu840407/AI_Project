"""YaYan_ASR_Mandarin — 漢語方言 ASR 包裝（SenseVoiceSmall）。"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Mandarin")

_MODEL = None


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    from funasr import AutoModel

    local_dir = model_path("YaYan_ASR_Mandarin")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Mandarin 不存在: {local_dir}")

    logger.info("載入 YaYan_ASR_Mandarin …")
    device = CONFIG["devices"]["asr_gpu"]
    _MODEL = AutoModel(
        model=str(local_dir),
        trust_remote_code=True,
        disable_update=True,
        device=device,
    )
    return _MODEL


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    """language_hint: zh/yue/wuu/cmn/auto"""
    model = _load()
    sv_lang = {"zh": "zh", "cmn": "zh", "yue": "yue", "wuu": "zh", "cdo": "zh"}.get(
        (language_hint or "auto").lower(), "auto"
    )
    res = model.generate(
        input=audio,
        cache={},
        language=sv_lang,
        use_itn=True,
        batch_size_s=60,
    )
    if not res:
        return ""
    text = res[0].get("text", "") if isinstance(res[0], dict) else str(res[0])
    return _strip_tags(text)


def _strip_tags(text: str) -> str:
    """移除 SenseVoice 的 <|...|> 控制標籤。"""
    import re
    return re.sub(r"<\|[^|]+\|>", "", text).strip()
