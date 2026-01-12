from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        v = float(s)
        if not math.isfinite(v):
            return None
        return float(v)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return None
        return int(float(s))
    except Exception:
        return None


def guess_trial_id_from_dir(trial_dir: Path) -> str:
    """
    Best-effort parse trial id from directory name:
      trainable_<exp>_<trialid>_<idx>_... -> <exp>_<trialid>
    """
    name = trial_dir.name
    if name.startswith("trainable_"):
        parts = name.split("_")
        # Example: trainable_d9a15_00057_57_...
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
    return ""


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}, got: {type(obj)}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    # Allow partially written files while a trial is running.
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except Exception:
        return records
    return records

