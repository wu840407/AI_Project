"""YaYan_ASR_Global — 通用多語 ASR（波斯語、烏爾都語、英語等）。"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Global")

_PIPE = None


def _load():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    from transformers import pipeline

    local_dir = model_path("YaYan_ASR_Global")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Global 不存在: {local_dir}")
    logger.info("載入 YaYan_ASR_Global …")
    device = CONFIG["devices"]["asr_gpu"]
    _PIPE = pipeline(
        "automatic-speech-recognition",
        model=str(local_dir),
        torch_dtype=torch.float16,
        device=device,
        model_kwargs={"local_files_only": True},
    )
    return _PIPE


_LANG_HINT = {
    "fa": "persian",
    "ur": "urdu",
    "en": "english",
    "ar": "arabic",
}


def transcribe(audio: np.ndarray, language_hint: Optional[str] = None) -> str:
    pipe = _load()
    sample_rate = CONFIG["audio"]["sample_rate"]
    gen_kwargs = {"task": "transcribe"}
    lang = _LANG_HINT.get((language_hint or "").lower())
    if lang:
        gen_kwargs["language"] = lang

    try:
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs=gen_kwargs,
            return_timestamps=True,
        )
    except Exception as e:
        logger.warning(f"YaYan_ASR_Global 識別失敗，回退無語言提示: {e}")
        out = pipe(
            {"raw": audio.astype(np.float32), "sampling_rate": sample_rate},
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True,
        )
    return out["text"].strip() if isinstance(out, dict) else str(out)
