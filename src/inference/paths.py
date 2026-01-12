from __future__ import annotations

import glob
import os
import re
from typing import List


def safe_slug(value: str) -> str:
    s = str(value or "").strip()
    if not s:
        return "model"
    # Replace path separators and other unsafe characters.
    s = s.replace(os.sep, "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("._-")
    return s or "model"


def resolve_path_best_effort(project_dir: str, p: str) -> str:
    """
    Resolve a user-provided path against PROJECT_DIR, with a small compatibility
    shim for packaged 'weights/' directories.
    """
    if not isinstance(p, str) or len(p.strip()) == 0:
        return ""
    p = p.strip()
    if os.path.isabs(p):
        return p
    # First: relative to project_dir
    cand = os.path.abspath(os.path.join(project_dir, p))
    if os.path.exists(cand):
        return cand
    # Compatibility: if a path starts with "weights/" but PROJECT_DIR already *is* the weights dir,
    # try stripping the prefix.
    if p.startswith("weights/") or p.startswith("weights" + os.sep):
        p2 = p.split("/", 1)[1] if "/" in p else p.split(os.sep, 1)[1]
        cand2 = os.path.abspath(os.path.join(project_dir, p2))
        if os.path.exists(cand2):
            return cand2
    return cand


def resolve_version_train_yaml(project_dir: str, version: str) -> str:
    """
    Resolve a per-version train.yaml snapshot (best effort).

    Search order:
      1) <PROJECT_DIR>/weights/configs/versions/<ver>/train.yaml   (repo-root running)
      2) <PROJECT_DIR>/configs/versions/<ver>/train.yaml           (running inside packaged weights dir)
      3) <PROJECT_DIR>/outputs/<ver>/train.yaml
      4) <PROJECT_DIR>/outputs/<ver>/fold_0/train.yaml
      5) <PROJECT_DIR>/configs/train.yaml                          (fallback)
    """
    ver = str(version or "").strip()
    candidates: List[str] = []
    if ver:
        candidates.extend(
            [
                os.path.join(project_dir, "weights", "configs", "versions", ver, "train.yaml"),
                os.path.join(project_dir, "configs", "versions", ver, "train.yaml"),
                os.path.join(project_dir, "outputs", ver, "train.yaml"),
                os.path.join(project_dir, "outputs", ver, "fold_0", "train.yaml"),
            ]
        )
    candidates.append(os.path.join(project_dir, "configs", "train.yaml"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Return the last fallback even if missing to preserve error context.
    return candidates[-1]


def resolve_version_head_base(project_dir: str, version: str) -> str:
    """
    Resolve the packaged head directory for a given version (best effort).

    Search order:
      1) <PROJECT_DIR>/weights/head/<ver>   (repo-root running)
      2) <PROJECT_DIR>/head/<ver>           (running inside packaged weights dir)
    """
    ver = str(version or "").strip()
    if not ver:
        return ""
    candidates = [
        os.path.join(project_dir, "weights", "head", ver),
        os.path.join(project_dir, "head", ver),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def find_dino_weights_in_dir(dir_path: str, backbone: str) -> str:
    """
    Find backbone weights file in a directory by known filename/patterns.
    Assumes the base filenames are stable; extension may be .pt or .pth.
    """
    if not (isinstance(dir_path, str) and dir_path and os.path.isdir(dir_path)):
        return ""
    bn2 = str(backbone or "").strip().lower()
    # Prefer exact known filenames first (fast + unambiguous).
    exact_bases: List[str] = []
    if bn2 in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
        exact_bases.append("dinov3_vit7b16_pretrain_lvd1689m-a955f4ea")
    elif bn2 == "dinov3_vits16":
        exact_bases.append("dinov3_vits16_pretrain_lvd1689m-08c60483")
    elif bn2 == "dinov3_vits16plus":
        exact_bases.append("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa")
    elif bn2 == "dinov3_vith16plus":
        exact_bases.append("dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5")
    elif bn2 == "dinov3_vitl16":
        exact_bases.append("dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd")
    for base in exact_bases:
        for ext in (".pt", ".pth"):
            cand = os.path.join(dir_path, base + ext)
            if os.path.isfile(cand):
                return os.path.abspath(cand)

    # Fallback: glob patterns (still scoped to the directory).
    patterns2: List[str] = []
    if bn2 in ("dinov3_vit7b16", "dinov3_vit7b", "vit7b16", "vit7b"):
        patterns2.extend(["dinov3_vit7b16_pretrain_*.pt", "dinov3_vit7b16_pretrain_*.pth"])
    elif bn2 == "dinov3_vits16":
        patterns2.extend(["dinov3_vits16_pretrain_*.pt", "dinov3_vits16_pretrain_*.pth"])
    elif bn2 == "dinov3_vits16plus":
        patterns2.extend(["dinov3_vits16plus_pretrain_*.pt", "dinov3_vits16plus_pretrain_*.pth"])
    elif bn2 == "dinov3_vith16plus":
        patterns2.extend(["dinov3_vith16plus_pretrain_*.pt", "dinov3_vith16plus_pretrain_*.pth"])
    elif bn2 == "dinov3_vitl16":
        patterns2.extend(["dinov3_vitl16_pretrain_*.pt", "dinov3_vitl16_pretrain_*.pth"])
    if bn2:
        patterns2.extend([f"{bn2}*.pt", f"{bn2}*.pth"])
    for pat in patterns2:
        try:
            for cand in sorted(glob.glob(os.path.join(dir_path, pat))):
                if os.path.isfile(cand):
                    return os.path.abspath(cand)
        except Exception:
            continue
    return ""


