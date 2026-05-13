"""YaYan_ASR_Dialect — Dolphin-CN-Dialect 包裝。

對應 Dolphin SDK 的 model(waveform, lang_sym, region_sym) API。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..config import CONFIG, model_path

logger = logging.getLogger("YaYan.ASR.Dolphin")

_MODEL = None

ROUTING_TO_DOLPHIN: dict = {
    "zh":    ("zh", "CN"),
    "cmn":   ("zh", "CN"),
    "yue":   ("yue", "CN"),
    "wuu":   ("wuu", "CN"),
    "wuu-sz":("wuu", "CN-SZ"),
    "wuu-nb":("wuu", "CN-NB"),
    "wuu-wz":("wuu", "CN-WZ"),
    "nan":   ("nan", "CN"),
    "nan-cs":("nan", "CN-CS"),
    "nan-hn":("nan", "CN-HN"),
    "cdo":   ("cdo", "CN"),
    "hak":   ("hak", "CN"),
    "hsn":   ("hsn", "CN"),
    "gan":   ("gan", "CN"),
    "cjy":   ("cjy", "CN"),
    "cmn-sw":("zh", "CN-SC"),       # ★ 西南官話 / 四川話 — 修正 region_sym
    "cmn-sd":("zh", "CN-SD"),
    "cmn-ne":("zh", "CN-NE"),
    "cmn-zy":("zh", "CN-ZY"),
    "cmn-wh":("zh", "CN-WH"),
    "cmn-xa":("zh", "CN-XA"),
    "cmn-lz":("zh", "CN-LZ"),
    "cmn-jh":("zh", "CN-JH"),
    "bo":    ("bo", "CN"),
    "ug":    ("ug", "CN"),
}


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class DolphinResult:
    text: str
    words: List[WordTimestamp]
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

    device = CONFIG["devices"]["asr_gpu"]
    logger.info(f"載入 YaYan_ASR_Dialect ({local_dir.name}) on {device} …")

    # Dolphin SDK 簽名：load_model(model_name, model_dir, device)
    # model_name 由本地檔名推斷（small.cn.pt → "small.cn"，small.pt → "small"）
    pt_files = list(local_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(
            f"{local_dir} 找不到 .pt 模型權重檔"
        )
    # 取最大的 .pt 檔當主模型（避免 optimizer.pt 之類）
    pt_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    model_name = pt_files[0].stem  # e.g. "small.cn" or "small"
    logger.info(f"使用 Dolphin model_name = {model_name}（檔案 {pt_files[0].name}）")

    _MODEL = dolphin.load_model(model_name, str(local_dir), device)
    return _MODEL


def transcribe(
    audio: np.ndarray,
    language_hint: Optional[str] = None,
    enable_word_timestamp: bool = True,
) -> DolphinResult:
    """執行 Dolphin ASR。

    audio: numpy float32 array (16kHz mono)
    language_hint: 例如 "zh" / "yue" / "nan" / "cdo" / "auto"
    """
    model = _load()

    lang_sym, region_sym = ROUTING_TO_DOLPHIN.get(
        (language_hint or "auto").lower(), (None, None)
    )

    # ★ numpy array → torch tensor（Dolphin SDK 內部需要）
    if isinstance(audio, np.ndarray):
        waveform = torch.from_numpy(audio).float()
    else:
        waveform = audio

    # ★ Dolphin SDK 用 model(waveform, **kwargs) 直接呼叫
    kwargs = {}
    if lang_sym:
        kwargs["lang_sym"] = lang_sym
    if region_sym:
        kwargs["region_sym"] = region_sym
    if enable_word_timestamp:
        # SDK 不同版本參數名不一樣，幾個都試
        kwargs["predict_time"] = True

    try:
        result = model(waveform, **kwargs)
    except TypeError:
        # 舊版 SDK 不支援 predict_time，去掉重試
        kwargs.pop("predict_time", None)
        try:
            result = model(waveform, **kwargs)
        except Exception as e:
            logger.error(f"Dolphin 推論失敗: {e}")
            return DolphinResult(text="", words=[])

    text = (getattr(result, "text", "") or "").strip()
    text = _strip_special_tokens(text)

    # 嘗試抽 word timestamp
    words: List[WordTimestamp] = []
    raw_words = getattr(result, "words", None) or getattr(result, "tokens", None)
    if raw_words:
        for w in raw_words:
            try:
                if isinstance(w, dict):
                    words.append(WordTimestamp(
                        word=str(w.get("word", w.get("text", ""))),
                        start=float(w.get("start", 0)),
                        end=float(w.get("end", 0)),
                    ))
                else:
                    words.append(WordTimestamp(
                        word=str(getattr(w, "word", getattr(w, "text", ""))),
                        start=float(getattr(w, "start", 0)),
                        end=float(getattr(w, "end", 0)),
                    ))
            except Exception:
                continue

    detected_lang = getattr(result, "language", "") or getattr(result, "lang", "") or ""
    detected_region = getattr(result, "region", "") or ""

    return DolphinResult(
        text=text, words=words,
        detected_lang=str(detected_lang),
        detected_region=str(detected_region),
    )


def _strip_special_tokens(text: str) -> str:
    text = re.sub(r"<\|[^|]*\|>", "", text)
    text = re.sub(r"<[a-z]{2,6}>", "", text)
    return text.strip()
