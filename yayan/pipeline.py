"""端到端流程 orchestrator：load audio → VAD → LID → ASR → LLM → OpenCC。"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import librosa
import numpy as np

from .config import CONFIG
from .llm import LlmClient, to_taiwan_traditional

logger = logging.getLogger("YaYan.Pipeline")


@dataclass
class Segment:
    start: float
    end: float
    speaker: str = "S0"
    raw_text: str = ""
    asr_alias: str = ""
    routing: str = ""


@dataclass
class TranscriptionResult:
    audio_path: str
    detected_language: str
    routing: str
    segments: List[Segment] = field(default_factory=list)
    raw_text: str = ""
    translated_text: str = ""

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "detected_language": self.detected_language,
            "routing": self.routing,
            "raw_text": self.raw_text,
            "translated_text": self.translated_text,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "speaker": s.speaker,
                    "raw_text": s.raw_text,
                    "asr_alias": s.asr_alias,
                }
                for s in self.segments
            ],
        }


_LLM: Optional[LlmClient] = None


def _get_llm() -> LlmClient:
    global _LLM
    if _LLM is None:
        _LLM = LlmClient(alias=CONFIG["llm"]["alias"])
    return _LLM


def warmup() -> None:
    """預先載入主要模型，縮短第一次推論延遲。"""
    _get_llm()
    logger.info("YaYan-AI v4.5 warmup 完成。")


def _load_audio(path: str) -> np.ndarray:
    sr = CONFIG["audio"]["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y.astype(np.float32)


def transcribe_audio(
    audio_path: str,
    language: str = "auto",
    use_vad: Optional[bool] = None,
    use_diarize: Optional[bool] = None,
) -> TranscriptionResult:
    """完整 pipeline：辨識方言 → 轉錄 → 翻譯 → 正體中文。"""
    audio_cfg = CONFIG["audio"]
    asr_cfg = CONFIG["asr"]
    use_vad = asr_cfg["enable_vad"] if use_vad is None else use_vad
    use_diarize = CONFIG["diarize"]["enabled"] if use_diarize is None else use_diarize

    audio = _load_audio(audio_path)
    sr = audio_cfg["sample_rate"]

    if language == "auto" and asr_cfg["enable_lid"]:
        from . import lid
        try:
            lid_audio = audio[: 30 * sr]
            routing, _conf = lid.detect(lid_audio, sample_rate=sr)
        except Exception as e:
            logger.warning(f"LID 失敗，回退預設路由: {e}")
            routing = "auto"
    else:
        routing = language

    if use_vad:
        from . import vad
        try:
            chunks = vad.split_speech(audio, sample_rate=sr)
        except Exception as e:
            logger.warning(f"VAD 失敗，整段直送: {e}")
            chunks = [(0.0, len(audio) / sr, audio)]
    else:
        chunks = [(0.0, len(audio) / sr, audio)]

    speakers = None
    if use_diarize:
        from . import diarize
        try:
            speakers = diarize.diarize(audio, sample_rate=sr)
        except Exception as e:
            logger.warning(f"Diarization 失敗，略過: {e}")

    from .asr import transcribe as asr_transcribe

    segments: List[Segment] = []
    for start, end, chunk in chunks:
        try:
            r = asr_transcribe(chunk, routing=routing, language_hint=routing)
        except Exception as e:
            logger.error(f"ASR 失敗 [{start:.1f}s-{end:.1f}s]: {e}")
            continue
        speaker_label = _label_speaker((start + end) / 2, speakers) if speakers else "S0"
        if r.text:
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    speaker=speaker_label,
                    raw_text=r.text,
                    asr_alias=r.asr_alias,
                    routing=r.routing,
                )
            )

    raw_text = "\n".join(
        f"[{s.speaker}] {s.raw_text}" if use_diarize else s.raw_text
        for s in segments
    )

    translated = ""
    if raw_text.strip():
        llm = _get_llm()
        translated = llm.translate(raw_text, source_language=routing)
        translated = to_taiwan_traditional(translated)

    return TranscriptionResult(
        audio_path=audio_path,
        detected_language=routing,
        routing=routing,
        segments=segments,
        raw_text=raw_text,
        translated_text=translated,
    )


def refine_with_user_edit(
    raw_text: str,
    user_edit: str,
    source_language: str = "zh",
) -> str:
    """對話修改回環：使用者編輯後重新潤飾，不重做 ASR。"""
    llm = _get_llm()
    refined = llm.refine(raw_text=raw_text, user_edit=user_edit, source_language=source_language)
    return to_taiwan_traditional(refined)


def _label_speaker(t: float, segments) -> str:
    if not segments:
        return "S0"
    for s, e, label in segments:
        if s <= t <= e:
            return label
    return "S0"
