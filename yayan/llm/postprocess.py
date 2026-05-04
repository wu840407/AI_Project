"""YaYan_TWConv — 強制台灣正體後處理（OpenCC s2twp）。"""
from __future__ import annotations

import logging
from typing import Optional

from ..config import CONFIG

logger = logging.getLogger("YaYan.TWConv")

_CONVERTER = None


def _get_converter():
    global _CONVERTER
    if _CONVERTER is not None:
        return _CONVERTER
    if not CONFIG["postprocess"]["use_opencc"]:
        return None
    try:
        from opencc import OpenCC
    except ImportError:
        logger.warning("opencc 未安裝，略過台灣正體轉換。")
        return None
    cfg_name = CONFIG["postprocess"]["opencc_config"]
    _CONVERTER = OpenCC(cfg_name.replace(".json", ""))
    return _CONVERTER


def to_taiwan_traditional(text: str) -> str:
    """簡體 → 台灣正體（含慣用詞替換）。輸入若已是正體，無實質改變。"""
    if not text:
        return text
    conv = _get_converter()
    if conv is None:
        return text
    try:
        return conv.convert(text)
    except Exception as e:
        logger.warning(f"OpenCC 轉換失敗: {e}")
        return text
