"""YaYan_LID — 語種識別（VoxLingua107 ECAPA-TDNN 包裝）。"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.LID")

_LID_MODEL = None

LANG_TO_ROUTING = {
    "zh": "zh", "cmn": "zh", "nan": "zh", "yue": "yue",
    "wuu": "wuu", "cdo": "cdo",
    "bo": "bo", "ug": "ug", "kk": "ug", "mn": "ug",
    "fa": "fa", "ur": "ur",
    "en": "en", "ja": "zh", "ko": "zh",
}


def _load() -> None:
    global _LID_MODEL
    if _LID_MODEL is not None:
        return
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier  # type: ignore

    local_dir = model_path("YaYan_LID")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_LID 不存在: {local_dir}")
    logger.info("載入 YaYan_LID …")
    device = CONFIG["devices"]["asr_gpu"]
    _LID_MODEL = EncoderClassifier.from_hparams(
        source=str(local_dir),
        savedir=str(local_dir),
        run_opts={"device": device},
    )


def detect(audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float]:
    """回傳 (routing_code, confidence)。routing_code 為 default.yaml 的 asr.routing 鍵。"""
    _load()
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    out = _LID_MODEL.classify_batch(audio_t)
    score = float(out[1].exp().max().item())
    label = out[3][0]
    iso639 = label.split(":")[0].strip().lower() if isinstance(label, str) else "auto"
    routing = LANG_TO_ROUTING.get(iso639, "auto")
    logger.info(f"YaYan_LID 偵測: {iso639} → routing={routing} (conf={score:.2f})")
    return routing, score
