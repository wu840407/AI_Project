"""YaYan_ASR_Dialect — Dolphin-CN-Dialect-Small 包裝。

支援 22 種中文方言 + 字級時間戳。
取代 v4.5 的 sensevoice.py + 舊版 dolphin-base。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Dolphin")

_MODEL = None

# routing key (default.yaml asr.routing) → Dolphin 的 (lang_sym, region_sym)
# 22 方言 + 通用語言對應
ROUTING_TO_DOLPHIN: dict = {
    # 漢語方言（cmn 系列）
    "zh":    ("zh", "CN"),
    "cmn":   ("zh", "CN"),
    "yue":   ("yue", "CN"),         # 粵語
    "wuu":   ("wuu", "CN"),         # 吳語（含上海話）
    "wuu-sz":("wuu", "CN-SZ"),      # 蘇州話
    "wuu-nb":("wuu", "CN-NB"),      # 寧波話
    "wuu-wz":("wuu", "CN-WZ"),      # 溫州話
    "nan":   ("nan", "CN"),         # 閩南語 / 台語
    "nan-cs":("nan", "CN-CS"),      # 潮汕話
    "nan-hn":("nan", "CN-HN"),      # 海南話
    "cdo":   ("cdo", "CN"),         # 閩東語 / 福州話
    "hak":   ("hak", "CN"),         # 客家話
    "hsn":   ("hsn", "CN"),         # 湘語 / 湖南話
    "gan":   ("gan", "CN"),         # 贛語 / 江西話
    "cjy":   ("cjy", "CN"),         # 晉語 / 山西話
    "cmn-sw":("zh", "CN-SW"),       # 四川話 / 西南官話
    "cmn-sd":("zh", "CN-SD"),       # 山東話
    "cmn-ne":("zh", "CN-NE"),       # 東北話
    "cmn-zy":("zh", "CN-ZY"),       # 河南話
    "cmn-wh":("zh", "CN-WH"),       # 武漢話
    "cmn-xa":("zh", "CN-XA"),       # 西安話
    "cmn-lz":("zh", "CN-LZ"),       # 蘭州話
    "cmn-jh":("zh", "CN-JH"),       # 南京話
    # 中亞 / 藏維
    "bo":    ("bo", "CN"),
    "ug":    ("ug", "CN"),
}


@dataclass
class WordTimestamp:
    """字級時間戳記。"""
    word: str
    start: float    # 相對於 chunk 開頭
    end: float


@dataclass
class DolphinResult:
    text: str
    words: List[WordTimestamp]      # 可能為空（短段沒做 timestamp）
    detected_lang: str = ""
    detected_region: str = ""


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import dolphin
    except ImportError as e:
        raise ImportError(
            "dolphin SDK 未安裝。請：pip install dataoceanai-dolphin"
        ) from e

    local_dir = model_path("YaYan_ASR_Dialect")
    if not local_dir.exists():
        raise FileNotFoundError(f"YaYan_ASR_Dialect 不存在: {local_dir}")

    logger.info(f"載入 YaYan_ASR_Dialect (Dolphin-CN-Dialect-Small) …")
    device = CONFIG["devices"]["asr_gpu"]

    # dolphin.load_model 預設會去 HF 抓；給 model_dir 走本地
    _MODEL = dolphin.load_model(
        "small.cn",
        model_dir=str(local_dir),
        device=device,
    )
    return _MODEL


def transcribe(
    audio: np.ndarray,
    language_hint: Optional[str] = None,
    enable_word_timestamp: bool = True,
) -> DolphinResult:
    """執行 Dolphin ASR。

    language_hint: 例如 "zh" / "yue" / "wuu" / "nan" / "cdo" / "auto"
    """
    import dolphin as _dolphin_pkg

    model = _load()

    lang_sym, region_sym = ROUTING_TO_DOLPHIN.get(
        (language_hint or "auto").lower(), (None, None)
    )

    kwargs = {}
    if lang_sym:
        kwargs["lang_sym"] = lang_sym
    if region_sym:
        kwargs["region_sym"] = region_sym
    if enable_word_timestamp:
        kwargs["word_timestamp"] = True

    # Dolphin SDK 接受 numpy array 或路徑
    try:
        result = _dolphin_pkg.transcribe(model, audio, **kwargs)
    except TypeError:
        # 老版 SDK 不支援 word_timestamp，降級
        kwargs.pop("word_timestamp", None)
        result = _dolphin_pkg.transcribe(model, audio, **kwargs)

    text = (getattr(result, "text", "") or "").strip()
    text = _strip_special_tokens(text)

    # 嘗試抽 word timestamp
    words: List[WordTimestamp] = []
    if hasattr(result, "words") and result.words:
        for w in result.words:
            try:
                words.append(WordTimestamp(
                    word=str(w.get("word", "")) if isinstance(w, dict) else str(w.word),
                    start=float(w.get("start", 0)) if isinstance(w, dict) else float(w.start),
                    end=float(w.get("end", 0)) if isinstance(w, dict) else float(w.end),
                ))
            except Exception:
                continue

    detected_lang = getattr(result, "language", "") or ""
    detected_region = getattr(result, "region", "") or ""

    return DolphinResult(
        text=text,
        words=words,
        detected_lang=detected_lang,
        detected_region=detected_region,
    )


def _strip_special_tokens(text: str) -> str:
    """移除 <|...|> 或 <xx> 等特殊 token。"""
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"<[a-z]{2,4}>", "", text)
    return text.strip()
