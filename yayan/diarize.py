"""YaYan_Diarize — 說話人分離（pyannote-3.1 包裝），預設關閉。"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.Diarize")

_PIPE = None


def is_available() -> bool:
    return model_path("YaYan_Diarize").exists()


def _load():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    if not is_available():
        raise FileNotFoundError(
            "YaYan_Diarize 模型未下載；如需啟用請於連網機器執行 download_models.sh --with-diarize"
        )
    from pyannote.audio import Pipeline
    import torch

    config_path = model_path("YaYan_Diarize") / "config.yaml"
    _PIPE = Pipeline.from_pretrained(str(config_path))
    device = CONFIG["devices"]["asr_gpu"]
    _PIPE.to(torch.device(device))
    return _PIPE


def diarize(
    audio: np.ndarray,
    sample_rate: int = 16000,
    min_speakers: int = None,
    max_speakers: int = None,
) -> List[Tuple[float, float, str]]:
    """回傳 [(start_sec, end_sec, speaker_label)]。"""
    cfg = CONFIG["diarize"]
    min_speakers = min_speakers or cfg["min_speakers"]
    max_speakers = max_speakers or cfg["max_speakers"]

    pipe = _load()
    import torch

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    diar = pipe(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    logger.info(f"YaYan_Diarize 分出 {len(set(s[2] for s in segments))} 位說話人。")
    return segments
