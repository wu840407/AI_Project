"""YaYan-AI v4.6 端到端流程。

v4.6 改進：
  - ASR 升級到 Dolphin-CN-Dialect-Small（22 方言）
  - 永遠逐段做 LID
  - 字級/段級時間戳，輸出格式：[A方 00:01-00:05] 你好
  - 說話人標籤 A/B/C/D/E（最多 5 人）
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np

from .config import CONFIG
from .llm import LlmClient, to_taiwan_traditional

logger = logging.getLogger("YaYan.Pipeline")

SPEAKER_LABELS = ["A", "B", "C", "D", "E"]


@dataclass
class Segment:
    start: float
    end: float
    speaker: str = "A"
    raw_text: str = ""
    asr_alias: str = ""
    routing: str = ""
    lid_conf: float = 0.0
    lid_method: str = "default"


@dataclass
class TranscriptionResult:
    audio_path: str
    detected_language: str
    routing: str
    segments: List[Segment] = field(default_factory=list)
    raw_text: str = ""
    translated_text: str = ""
    language_breakdown: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "audio_path": self.audio_path,
            "detected_language": self.detected_language,
            "routing": self.routing,
            "language_breakdown": self.language_breakdown,
            "raw_text": self.raw_text,
            "translated_text": self.translated_text,
            "segments": [
                {
                    "start": s.start, "end": s.end, "speaker": s.speaker,
                    "raw_text": s.raw_text, "asr_alias": s.asr_alias,
                    "routing": s.routing, "lid_conf": s.lid_conf,
                    "lid_method": s.lid_method,
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
    _get_llm()
    logger.info("YaYan-AI v4.6 warmup 完成。")


def _load_audio(path: str) -> np.ndarray:
    sr = CONFIG["audio"]["sample_rate"]
    y, _ = librosa.load(path, sr=sr, mono=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def _normalize_speakers(
    raw_segments: List[Tuple[float, float, str]],
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    next_idx = 0
    for _, _, label in raw_segments:
        if label not in mapping:
            if next_idx < len(SPEAKER_LABELS):
                mapping[label] = SPEAKER_LABELS[next_idx]
                next_idx += 1
            else:
                mapping[label] = SPEAKER_LABELS[-1]
    return mapping


def _label_speaker_at(
    t: float,
    raw_segments: List[Tuple[float, float, str]],
    mapping: Dict[str, str],
) -> str:
    if not raw_segments:
        return "A"
    for s, e, label in raw_segments:
        if s <= t <= e:
            return mapping.get(label, "A")
    return "A"


def _fmt_time(sec: float) -> str:
    """1.23 → '00:01' / 65.4 → '01:05' / 3661 → '1:01:01'"""
    sec = int(round(sec))
    h, remain = divmod(sec, 3600)
    m, s = divmod(remain, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _dynamic_lid_threshold(seg_dur_sec: float) -> float:
    if seg_dur_sec < 2.0:
        return 0.40
    elif seg_dur_sec < 5.0:
        return 0.50
    else:
        return 0.60


def _vote_routing(
    candidates: List[Tuple[str, float]],
    fallback: str,
) -> str:
    valid = [(r, c) for r, c in candidates if c > 0.3 and r != "auto"]
    if not valid:
        return fallback
    weighted: Counter = Counter()
    for r, c in valid:
        weighted[r] += c
    return weighted.most_common(1)[0][0]


def transcribe_audio(
    audio_path: str,
    language: str = "auto",
    use_vad: Optional[bool] = None,
    use_diarize: Optional[bool] = None,
) -> TranscriptionResult:
    audio_cfg = CONFIG["audio"]
    asr_cfg = CONFIG["asr"]
    use_vad = asr_cfg["enable_vad"] if use_vad is None else use_vad
    use_diarize = CONFIG["diarize"]["enabled"] if use_diarize is None else use_diarize

    audio = _load_audio(audio_path)
    sr = audio_cfg["sample_rate"]

    if audio.size == 0:
        logger.warning(f"音檔為空: {audio_path}")
        return TranscriptionResult(
            audio_path=audio_path,
            detected_language=language, routing=language,
        )

    user_hint = language
    
    # ---- VAD ----
    if use_vad:
        from . import vad
        try:
            chunks = vad.split_speech(audio, sample_rate=sr)
        except Exception as e:
            logger.warning(f"VAD 失敗，整段直送: {e}")
            chunks = [(0.0, len(audio) / sr, audio)]
    else:
        chunks = [(0.0, len(audio) / sr, audio)]

    # ---- Diarization ----
    raw_speakers: List[Tuple[float, float, str]] = []
    speaker_mapping: Dict[str, str] = {}
    if use_diarize:
        from . import diarize
        try:
            raw_speakers = diarize.diarize(audio, sample_rate=sr)
            speaker_mapping = _normalize_speakers(raw_speakers)
            logger.info(
                f"Diarize 偵測 {len(speaker_mapping)} 位 → "
                f"{', '.join(sorted(set(speaker_mapping.values())))}"
            )
        except Exception as e:
            logger.warning(f"Diarization 失敗，略過: {e}")

    # ---- 逐段 LID ----
    from . import lid as _lid_mod
    enable_lid = asr_cfg["enable_lid"]
    seg_routings: List[Tuple[str, float, str]] = []

    if enable_lid:
        logger.info(f"逐段 LID（{len(chunks)} 段）...")
        for i, (start, end, chunk) in enumerate(chunks):
            seg_dur = end - start
            if seg_dur < 0.5:
                seg_routings.append(("auto", 0.0, "too_short"))
                continue
            try:
                rt, conf = _lid_mod.detect(chunk, sample_rate=sr)
                threshold = _dynamic_lid_threshold(seg_dur)
                if conf >= threshold:
                    seg_routings.append((rt, conf, "lid"))
                else:
                    seg_routings.append(("auto", conf, "low_conf"))
            except Exception as e:
                logger.debug(f"段 {i} LID 失敗: {e}")
                seg_routings.append(("auto", 0.0, "lid_error"))
    else:
        seg_routings = [(user_hint, 1.0, "user_hint")] * len(chunks)

    # ---- 補齊 auto 段（鄰近投票）----
    fallback_routing = user_hint if user_hint != "auto" else "zh"
    for i, (rt, conf, method) in enumerate(seg_routings):
        if rt != "auto":
            continue
        window = []
        for j in range(max(0, i - 3), min(len(seg_routings), i + 4)):
            if j == i:
                continue
            r, c, _ = seg_routings[j]
            window.append((r, c))
        voted = _vote_routing(window, fallback_routing)
        seg_routings[i] = (voted, 0.0,
                          "neighbor_vote" if voted != fallback_routing else "fallback")

    # ---- 同說話人連貫性 ----
    if use_diarize and raw_speakers:
        last_speaker = None
        last_routing = None
        for i, (start, end, _chunk) in enumerate(chunks):
            mid_t = (start + end) / 2
            speaker = _label_speaker_at(mid_t, raw_speakers, speaker_mapping)
            current_routing, conf, method = seg_routings[i]
            if speaker == last_speaker and method in ("low_conf", "lid_error", "fallback"):
                if last_routing:
                    seg_routings[i] = (last_routing, conf, "speaker_inherit")
            if method == "lid" and conf >= 0.6:
                last_speaker = speaker
                last_routing = current_routing

    # ---- 執行 ASR ----
    from .asr import transcribe as asr_transcribe

    segments: List[Segment] = []
    lang_counter: Counter = Counter()

    for i, (start, end, chunk) in enumerate(chunks):
        routing, conf, method = seg_routings[i]
        try:
            r = asr_transcribe(
                chunk, routing=routing, language_hint=routing,
                chunk_start_sec=start,  # 給 word timestamp 用
            )
        except Exception as e:
            logger.error(f"ASR 失敗 [{start:.1f}s-{end:.1f}s] routing={routing}: {e}")
            continue

        speaker_label = _label_speaker_at(
            (start + end) / 2, raw_speakers, speaker_mapping
        ) if use_diarize else "A"

        if r.text:
            text_tw = to_taiwan_traditional(r.text)
            segments.append(Segment(
                start=start, end=end,
                speaker=speaker_label,
                raw_text=text_tw,
                asr_alias=r.asr_alias,
                routing=routing,
                lid_conf=conf,
                lid_method=method,
            ))
            lang_counter[routing] += 1

    # ---- 統計 ----
    if lang_counter:
        dominant_lang = lang_counter.most_common(1)[0][0]
    else:
        dominant_lang = user_hint if user_hint != "auto" else "zh"
    logger.info(f"語言分布: {dict(lang_counter)} | 主要: {dominant_lang}")

    # ---- 組合 raw_text（v4.6 新格式：[A方 00:01-00:05] 內容）----
    raw_text_lines = []
    for s in segments:
        t1 = _fmt_time(s.start)
        t2 = _fmt_time(s.end)
        if use_diarize:
            prefix = f"[{s.speaker}方 {t1}-{t2}]"
        else:
            prefix = f"[{t1}-{t2}]"
        raw_text_lines.append(f"{prefix} {s.raw_text}")
    raw_text = "\n".join(raw_text_lines).strip()

    # ---- LLM 翻譯 ----
    translated = ""
    if raw_text:
        try:
            llm = _get_llm()
            lang_summary = ", ".join(f"{k}({v}段)" for k, v in lang_counter.most_common(3))
            translated = llm.translate(
                raw_text,
                source_language=f"{dominant_lang} (混合: {lang_summary})",
            )
        except Exception as e:
            logger.exception(f"LLM 翻譯失敗: {e}")
            translated = raw_text
        translated = to_taiwan_traditional(translated)

    return TranscriptionResult(
        audio_path=audio_path,
        detected_language=dominant_lang,
        routing=dominant_lang,
        segments=segments,
        raw_text=raw_text,
        translated_text=translated,
        language_breakdown=dict(lang_counter),
    )


def refine_with_user_edit(
    raw_text: str, user_edit: str, source_language: str = "zh",
) -> str:
    if not user_edit or not user_edit.strip():
        return ""
    raw_text = (raw_text or user_edit).strip()
    user_edit = user_edit.strip()

    try:
        llm = _get_llm()
        refined = llm.refine(
            raw_text=raw_text, user_edit=user_edit,
            source_language=source_language,
        )
    except Exception as e:
        logger.exception(f"LLM refine 失敗: {e}")
        return to_taiwan_traditional(user_edit)

    return to_taiwan_traditional(refined)
