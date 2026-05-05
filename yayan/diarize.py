"""YaYan_Diarize — 離線環境下載入 pyannote speaker-diarization-3.1。

關鍵：pyannote 3.1 的 pipeline YAML 內預設引用 HF repo ID
   segmentation: pyannote/segmentation-3.0
   embedding:    pyannote/wespeaker-voxceleb-resnet34-LM
離線環境會 404。本模組會把 YAML 改寫成指向本地路徑後再載入。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import yaml

from .config import CONFIG, model_path

logger = logging.getLogger("YaYan.Diarize")

_PIPELINE = None


def _ensure_offline_yaml(pipeline_dir: Path,
                         seg_dir: Path,
                         embed_dir: Path) -> Path:
    """在 pipeline_dir 旁邊產生一份指向本地的 config.yaml，並回傳該路徑。"""
    src = pipeline_dir / "config.yaml"
    if not src.exists():
        raise FileNotFoundError(f"找不到 pyannote pipeline 設定: {src}")

    patched = pipeline_dir / "config.offline.yaml"
    if patched.exists():
        return patched

    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    params = cfg.setdefault("pipeline", {}).setdefault("params", {})

    # segmentation 要指向 .bin / .ckpt 檔，不是目錄
    seg_ckpt = _find_weight_file(seg_dir, ("pytorch_model.bin", "model.safetensors"))
    embed_ckpt = _find_weight_file(embed_dir, ("pytorch_model.bin", "model.safetensors"))

    params["segmentation"] = str(seg_ckpt)
    params["embedding"] = str(embed_ckpt)

    with patched.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    logger.info(f"已產生離線 pyannote 設定: {patched}")
    return patched


def _find_weight_file(d: Path, candidates: Tuple[str, ...]) -> Path:
    for name in candidates:
        p = d / name
        if p.exists():
            return p
    # 退一步：找任何 .bin / .ckpt
    for ext in (".bin", ".ckpt", ".safetensors"):
        for p in d.glob(f"*{ext}"):
            return p
    raise FileNotFoundError(f"在 {d} 找不到模型權重檔（{candidates}）")


def _load_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise ImportError(
            "pyannote.audio 未安裝。請：pip install 'pyannote.audio>=3.1'"
        ) from e

    pipeline_dir = model_path("YaYan_Diarize")
    seg_dir = model_path("YaYan_Diarize_Seg")
    embed_dir = model_path("YaYan_Diarize_Embed")

    for d, name in [(pipeline_dir, "YaYan_Diarize"),
                    (seg_dir, "YaYan_Diarize_Seg"),
                    (embed_dir, "YaYan_Diarize_Embed")]:
        if not d.exists() or not any(d.iterdir()):
            raise FileNotFoundError(
                f"{name} 不存在或為空: {d}\n"
                f"請執行：HF_TOKEN=hf_xxx bash scripts/download_models.sh --with-diarize"
            )

    yaml_path = _ensure_offline_yaml(pipeline_dir, seg_dir, embed_dir)
    logger.info(f"載入 pyannote pipeline: {yaml_path}")
    _PIPELINE = Pipeline.from_pretrained(str(yaml_path))

    # 把 pipeline 移到指定 GPU
    try:
        import torch
        device = torch.device(CONFIG["devices"]["asr_gpu"])
        _PIPELINE.to(device)
        logger.info(f"pyannote pipeline → {device}")
    except Exception as e:
        logger.warning(f"無法將 pyannote pipeline 移到 GPU: {e}（將用 CPU）")

    return _PIPELINE


def diarize(audio: np.ndarray, sample_rate: int = 16000
            ) -> List[Tuple[float, float, str]]:
    """回傳 [(start_sec, end_sec, speaker_label), ...]。"""
    cfg = CONFIG["diarize"]
    pipeline = _load_pipeline()

    import torch
    waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)

    annotation = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        min_speakers=cfg.get("min_speakers", 1),
        max_speakers=cfg.get("max_speakers", 4),
    )

    segments: List[Tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append((float(turn.start), float(turn.end), str(speaker)))
    return segments
