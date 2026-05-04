#!/usr/bin/env python3
"""驗證所有 YaYan-AI v4.5 模型已正確放置在 models_root，可離線載入。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yayan.config import CONFIG, ALIASES, model_path  # noqa: E402

REQUIRED = [
    "YaYan_Reasoner",
    "YaYan_ASR_Mandarin",
    "YaYan_ASR_Eastern",
    "YaYan_ASR_Global",
    "YaYan_VAD",
    "YaYan_LID",
]
OPTIONAL = ["YaYan_Diarize"]


def check(alias: str, required: bool) -> bool:
    p = model_path(alias)
    if not p.exists():
        msg = "❌ 缺失" if required else "⚠️ 未下載（可選）"
        print(f"  {msg}: {alias} -> {p}")
        return not required
    files = list(p.iterdir())
    if not files:
        print(f"  ❌ 空資料夾: {alias} -> {p}")
        return not required
    size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"  ✅ {alias}  ({size_mb:.1f} MB)  {p}")
    return True


def main():
    print("=" * 60)
    print(f"  YaYan-AI v4.5 模型檢查")
    print(f"  models_root: {CONFIG['paths']['models_root']}")
    print("=" * 60)

    print("\n[必要]")
    ok = all(check(a, True) for a in REQUIRED)
    print("\n[可選]")
    for a in OPTIONAL:
        check(a, False)

    print("\n" + "=" * 60)
    if ok:
        print("✅ 所有必要模型已就緒，可啟動 app_rtx6000.py")
        sys.exit(0)
    else:
        print("❌ 有必要模型缺失，請先執行 scripts/download_models.sh")
        sys.exit(1)


if __name__ == "__main__":
    main()
