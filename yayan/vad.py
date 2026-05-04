"""YaYan_VAD — 語音活動偵測（Silero-VAD 包裝），所有外露名稱以 YaYan_VAD 為主。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.VAD")

_VAD_MODEL = None
_VAD_UTILS = None


def _load() -> None:
    global _VAD_MODEL, _VAD_UTILS
    if _VAD_MODEL is not None:
        return
    local_dir = model_path("YaYan_VAD")
    if not local_dir.exists():
        raise FileNotFoundError(
            f"YaYan_VAD 模型目錄不存在: {local_dir}\n"
            "請先在連網機器執行 scripts/download_models.sh"
        )
    logger.info("載入 YaYan_VAD …")
    _VAD_MODEL, _VAD_UTILS = torch.hub.load(
        repo_or_dir=str(local_dir),
        model="silero_vad",
        source="local",
        trust_repo=True,
    )


def split_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = None,
    max_chunk_seconds: float = None,
    min_chunk_seconds: float = None,
) -> List[Tuple[float, float, np.ndarray]]:
    """切分語音段落。回傳 [(start_sec, end_sec, chunk_array)]。"""
    _load()
    audio_cfg = CONFIG["audio"]
    threshold = threshold or audio_cfg["vad_threshold"]
    max_chunk_seconds = max_chunk_seconds or audio_cfg["max_chunk_seconds"]
    min_chunk_seconds = min_chunk_seconds or audio_cfg["min_chunk_seconds"]

    get_speech_timestamps = _VAD_UTILS[0]
    audio_t = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(
        audio_t,
        _VAD_MODEL,
        sampling_rate=sample_rate,
        threshold=threshold,
        max_speech_duration_s=max_chunk_seconds,
        min_speech_duration_ms=int(min_chunk_seconds * 1000),
    )

    if not timestamps:
        logger.warning("VAD 未偵測到語音段落，回傳整段。")
        return [(0.0, len(audio) / sample_rate, audio)]

    chunks: List[Tuple[float, float, np.ndarray]] = []
    pad = int(audio_cfg.get("pad_seconds", 0.2) * sample_rate)
    for ts in timestamps:
        start = max(0, ts["start"] - pad)
        end = min(len(audio), ts["end"] + pad)
        chunks.append(
            (start / sample_rate, end / sample_rate, audio[start:end])
        )
    logger.info(f"VAD 切出 {len(chunks)} 段語音。")
    return chunks
