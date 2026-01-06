"""
Nano Banana Pro augmentation tool for CSIRO training images (Gemini / OpenRouter).

Key features:
  - YAML-configured augmentation types + prompts
  - API key read from a file
    - Gemini: default `gemini.secret`
    - OpenRouter: default `openrouter.secret`
  - Saves generated images under data/nano_banana_pro/train by default
  - Logging + resumable runs (skip finished image_id + aug_name pairs)

This tool is intentionally standalone: the training pipeline consumes the generated
images via a manifest file and does NOT require Gemini dependencies at runtime.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import pandas as pd
import yaml
from loguru import logger
from PIL import Image


def resolve_repo_root() -> Path:
    """
    Resolve the project root directory that contains both `configs/` and `src/`.
    Mirrors the helper in train.py so this tool works from packaged locations too.
    """
    here = Path(__file__).resolve().parent
    # tools/nano_banana_pro -> repo root is two levels up
    candidates = [
        here,
        here.parent,
        here.parent.parent,
        here.parent.parent.parent,
    ]
    for c in candidates:
        if (c / "configs").is_dir() and (c / "src").is_dir():
            return c
    return here


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        # Some secret files may be ASCII-only without UTF-8 BOM; fall back.
        return path.read_text().strip()


def _truncate_text(s: str, max_chars: int = 2000) -> str:
    txt = str(s or "")
    if max_chars <= 0:
        return ""
    if len(txt) <= max_chars:
        return txt
    return txt[:max_chars] + f"...(truncated, {len(txt) - max_chars} more chars)"


def _guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    # Fallback: most CSIRO images are JPEG
    return "image/jpeg"


def _http_post_json(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: int,
) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib_request.urlopen(req, timeout=int(timeout_s)) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        # HTTPError is also a file-like object; read the response body for debugging.
        body_bytes: bytes = b""
        try:
            body_bytes = e.read() or b""
        except Exception:
            body_bytes = b""
        try:
            body_text = body_bytes.decode("utf-8", errors="replace")
        except Exception:
            body_text = str(body_bytes)

        request_id = None
        try:
            request_id = e.headers.get("x-request-id") or e.headers.get("X-Request-Id")
        except Exception:
            request_id = None

        rid = f" request_id={request_id}" if request_id else ""
        raise RuntimeError(
            f"HTTP {getattr(e, 'code', '?')} {getattr(e, 'reason', '')}{rid} "
            f"url={url} payload_bytes={len(data)} body={_truncate_text(body_text, 4000)}"
        ) from e


def _extract_first_image_bytes(resp_json: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Return (image_bytes, error_message). error_message is None on success.
    """
    try:
        candidates = resp_json.get("candidates") or []
        if not candidates:
            return None, "No candidates in response"
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        for p in parts:
            inline = p.get("inlineData") or p.get("inline_data")
            if inline and isinstance(inline, dict):
                b64 = inline.get("data")
                if not b64:
                    continue
                try:
                    return base64.b64decode(b64), None
                except Exception as e:
                    return None, f"Failed to base64-decode image data: {e}"
        return None, "No inlineData image part found in response"
    except Exception as e:
        return None, f"Unexpected response structure: {e}"


def _data_url_to_bytes(data_url: str) -> Tuple[Optional[bytes], Optional[str]]:
    """
    Decode a base64 data URL like:
      data:image/png;base64,AAAA...
    Returns (bytes, error_message).
    """
    s = str(data_url or "").strip()
    if not s:
        return None, "Empty image url"
    if s.startswith("data:"):
        if "base64," not in s:
            return None, "Unsupported data URL (missing base64,)"
        b64 = s.split("base64,", 1)[1]
        if not b64:
            return None, "Empty base64 payload in data URL"
        try:
            return base64.b64decode(b64), None
        except Exception as e:
            return None, f"Failed to base64-decode data URL: {e}"
    # Fallback: sometimes APIs may return raw base64 without the data: prefix.
    try:
        return base64.b64decode(s), None
    except Exception:
        return None, "Unsupported image url format (expected data:...;base64,...)"


def _extract_first_image_bytes_openrouter(resp_json: Dict[str, Any]) -> Tuple[Optional[bytes], Optional[str]]:
    """
    OpenRouter image-generation response format (per docs):
      choices[0].message.images[0].image_url.url = "data:image/png;base64,..."
    Returns (image_bytes, error_message).
    """
    try:
        # Surface API errors if present
        if isinstance(resp_json.get("error"), dict):
            msg = resp_json["error"].get("message") or resp_json["error"].get("error") or resp_json["error"]
            return None, f"OpenRouter error: {msg}"

        choices = resp_json.get("choices") or []
        if not choices:
            return None, "No choices in response"
        message = (choices[0] or {}).get("message") or {}
        images = message.get("images") or []
        if not images:
            # Some models may return nothing if modalities are misconfigured.
            return None, "No images in response message (check modalities/model)"
        first = images[0] or {}
        url = None
        if isinstance(first, dict):
            # snake_case
            if isinstance(first.get("image_url"), dict):
                url = first["image_url"].get("url")
            # camelCase (TypeScript docs show imageUrl)
            if url is None and isinstance(first.get("imageUrl"), dict):
                url = first["imageUrl"].get("url")
            # rare fallback
            if url is None:
                url = first.get("url")
        return _data_url_to_bytes(str(url or ""))
    except Exception as e:
        return None, f"Unexpected OpenRouter response structure: {e}"


def _pil_resample(name: str):
    n = str(name or "").strip().lower()
    try:
        resampling = Image.Resampling  # Pillow>=9
        mapping = {
            "nearest": resampling.NEAREST,
            "bilinear": resampling.BILINEAR,
            "bicubic": resampling.BICUBIC,
            "lanczos": resampling.LANCZOS,
        }
        return mapping.get(n, resampling.BICUBIC)
    except Exception:
        mapping = {
            "nearest": getattr(Image, "NEAREST", 0),
            "bilinear": getattr(Image, "BILINEAR", 2),
            "bicubic": getattr(Image, "BICUBIC", 3),
            "lanczos": getattr(Image, "LANCZOS", 1),
        }
        return mapping.get(n, getattr(Image, "BICUBIC", 3))


def _pad_or_crop_to_aspect(
    img: Image.Image,
    *,
    target_aspect: float,
    mode: str,
    pad_mode: str,
    pad_color: Tuple[int, int, int],
) -> Image.Image:
    """
    Ensure output image has width/height == target_aspect (exactly, via integer dims),
    using either padding or center-cropping.
    """
    if target_aspect <= 0:
        return img
    img = img.convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    cur = float(w) / float(h)
    if abs(cur - target_aspect) < 1e-9:
        return img

    m = str(mode or "pad").strip().lower()
    if m not in ("pad", "crop"):
        m = "pad"

    if m == "crop":
        # Center crop to target aspect
        if cur > target_aspect:
            # Too wide => crop width
            new_w = int(round(target_aspect * h))
            new_w = max(1, min(new_w, w))
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        # Too tall => crop height
        new_h = int(round(w / target_aspect))
        new_h = max(1, min(new_h, h))
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

    # Pad to target aspect (label-preserving; no content removed)
    if cur > target_aspect:
        # Too wide => add height
        new_h = int(round(w / target_aspect))
        new_h = max(h, new_h)
        new_w = w
        pad_top = (new_h - h) // 2
        pad_bottom = new_h - h - pad_top
        pad_left = pad_right = 0
    else:
        # Too tall => add width
        new_w = int(round(target_aspect * h))
        new_w = max(w, new_w)
        new_h = h
        pad_left = (new_w - w) // 2
        pad_right = new_w - w - pad_left
        pad_top = pad_bottom = 0

    p = str(pad_mode or "reflect").strip().lower()
    if p in ("reflect", "edge"):
        # Use numpy padding for non-constant padding modes.
        import numpy as np

        arr = np.asarray(img)
        arr2 = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=p,
        )
        return Image.fromarray(arr2.astype("uint8"), mode="RGB")

    # Constant color padding
    canvas = Image.new("RGB", (new_w, new_h), color=tuple(pad_color))
    canvas.paste(img, (pad_left, pad_top))
    return canvas


def _save_image_with_ext(
    img: Image.Image,
    *,
    out_path: Path,
    output_ext: str,
    png_compress_level: int = 6,
    jpeg_quality: int = 95,
    webp_quality: int = 90,
) -> None:
    ext = str(output_ext or "png").strip().lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        img.save(str(out_path), format="JPEG", quality=int(jpeg_quality))
        return
    if ext == "webp":
        img.save(str(out_path), format="WEBP", quality=int(webp_quality))
        return
    # default: png
    img.save(str(out_path), format="PNG", compress_level=int(png_compress_level))


def _postprocess_and_write(
    img_bytes: bytes,
    *,
    out_path: Path,
    output_ext: str,
    post_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decode model output -> optional aspect/size enforcement -> save with correct extension.
    Returns (width, height) of the written image.
    Falls back to raw bytes write if PIL decode fails.
    """
    try:
        img = Image.open(BytesIO(img_bytes))
        img = img.convert("RGB")
    except Exception:
        out_path.write_bytes(img_bytes)
        return {
            "orig_w": 0,
            "orig_h": 0,
            "aspect_w": 0,
            "aspect_h": 0,
            "out_w": 0,
            "out_h": 0,
            "pad_area_px": 0,
            "area_scale": 1.0,
        }

    # Aspect ratio enforcement (optional)
    orig_w, orig_h = img.size
    aspect_w, aspect_h = orig_w, orig_h
    pad_area_px = 0
    area_scale = 1.0
    enforce_aspect = post_cfg.get("enforce_aspect_ratio", None)
    if enforce_aspect not in (None, "", "null", False):
        try:
            target_aspect = float(enforce_aspect)
        except Exception:
            target_aspect = 0.0
        if target_aspect > 0:
            before_w, before_h = img.size
            img = _pad_or_crop_to_aspect(
                img,
                target_aspect=target_aspect,
                mode=str(post_cfg.get("mode", "pad")),
                pad_mode=str(post_cfg.get("pad_mode", "reflect")),
                pad_color=tuple(post_cfg.get("pad_color", [0, 0, 0])),
            )
            aspect_w, aspect_h = img.size
            before_area = int(before_w) * int(before_h) if (before_w > 0 and before_h > 0) else 0
            after_area = int(aspect_w) * int(aspect_h) if (aspect_w > 0 and aspect_h > 0) else 0
            if before_area > 0 and after_area > 0:
                area_scale = float(after_area) / float(before_area)
                # "padding area" only counts added pixels (pad mode). For crop mode, this stays 0.
                if str(post_cfg.get("mode", "pad")).strip().lower() == "pad":
                    pad_area_px = max(0, int(after_area - before_area))

    # Exact output size (optional) — YAML uses [W, H]
    target_size = post_cfg.get("target_size", None)
    if target_size not in (None, "", "null", False):
        try:
            if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                tw, th = int(target_size[0]), int(target_size[1])
                if tw > 0 and th > 0:
                    img = img.resize((tw, th), resample=_pil_resample(post_cfg.get("resample", "bicubic")))
        except Exception:
            pass

    # Save with desired extension
    _save_image_with_ext(
        img,
        out_path=out_path,
        output_ext=output_ext,
        png_compress_level=int(post_cfg.get("png_compress_level", 6)),
        jpeg_quality=int(post_cfg.get("jpeg_quality", 95)),
        webp_quality=int(post_cfg.get("webp_quality", 90)),
    )
    w, h = img.size
    return {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "aspect_w": int(aspect_w),
        "aspect_h": int(aspect_h),
        "out_w": int(w),
        "out_h": int(h),
        "pad_area_px": int(pad_area_px),
        "area_scale": float(area_scale if area_scale > 0 else 1.0),
    }


def _format_seconds(s: float) -> str:
    try:
        s = float(s)
    except Exception:
        return "?"
    if s < 0:
        s = 0.0
    # keep it compact but readable
    if s < 60:
        return f"{s:.2f}s"
    m = int(s // 60)
    sec = s - 60 * m
    if m < 60:
        return f"{m}m{sec:05.2f}s"
    h = int(m // 60)
    mm = m - 60 * h
    return f"{h}h{mm:02d}m{sec:05.2f}s"


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * float(done) / float(total)


def _run_gemini_job(
    *,
    base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    src_mime: str,
    src_b64: str,
    prompt: str,
    image_cfg: Optional[Dict[str, Any]],
    output_ext: str,
    post_cfg: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    """
    Worker function (thread-safe): call Gemini and write output image.
    Returns a dict with:
      - status: "success" | "failed"
      - attempts: int
      - elapsed_s: float
      - orig_w/orig_h/aspect_w/aspect_h/out_w/out_h/pad_area_px/area_scale
      - error: str (on failure)
    """
    t0 = time.perf_counter()
    attempt = 0
    last_err: str = ""
    while attempt < int(max_retries):
        attempt += 1
        try:
            payload: Dict[str, Any] = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": src_mime,
                                    "data": src_b64,
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                },
            }
            if isinstance(image_cfg, dict) and image_cfg:
                payload["generationConfig"]["imageConfig"] = dict(image_cfg)

            resp_json = _http_post_json(
                base_url, headers=headers, payload=payload, timeout_s=int(timeout_s)
            )
            img_bytes, err = _extract_first_image_bytes(resp_json)
            if err:
                raise RuntimeError(err)
            assert img_bytes is not None
            meta = _postprocess_and_write(
                img_bytes,
                out_path=out_path,
                output_ext=output_ext,
                post_cfg=post_cfg,
            )
            return {
                "status": "success",
                "attempts": int(attempt),
                "elapsed_s": float(time.perf_counter() - t0),
                **dict(meta or {}),
                "error": "",
            }
        except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
            last_err = str(e)
            if attempt >= int(max_retries):
                return {
                    "status": "failed",
                    "attempts": int(attempt),
                    "elapsed_s": float(time.perf_counter() - t0),
                    "orig_w": 0,
                    "orig_h": 0,
                    "aspect_w": 0,
                    "aspect_h": 0,
                    "out_w": 0,
                    "out_h": 0,
                    "pad_area_px": 0,
                    "area_scale": 1.0,
                    "error": last_err,
                }
            time.sleep(float(retry_backoff_s) * float(attempt))
        except Exception as e:
            last_err = str(e)
            return {
                "status": "failed",
                "attempts": int(attempt),
                "elapsed_s": float(time.perf_counter() - t0),
                "orig_w": 0,
                "orig_h": 0,
                "aspect_w": 0,
                "aspect_h": 0,
                "out_w": 0,
                "out_h": 0,
                "pad_area_px": 0,
                "area_scale": 1.0,
                "error": last_err,
            }
    return {
        "status": "failed",
        "attempts": int(attempt),
        "elapsed_s": float(time.perf_counter() - t0),
        "orig_w": 0,
        "orig_h": 0,
        "aspect_w": 0,
        "aspect_h": 0,
        "out_w": 0,
        "out_h": 0,
        "pad_area_px": 0,
        "area_scale": 1.0,
        "error": last_err or "unknown error",
    }


def _run_openrouter_job(
    *,
    base_url: str,
    headers: Dict[str, str],
    timeout_s: int,
    max_retries: int,
    retry_backoff_s: float,
    model: str,
    modalities: Sequence[str],
    src_mime: str,
    src_b64: str,
    prompt: str,
    image_cfg: Optional[Dict[str, Any]],
    output_ext: str,
    post_cfg: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    """
    Worker function (thread-safe): call OpenRouter /chat/completions and write output image.
    Uses multimodal input (source image + text prompt) and expects an image in:
      choices[0].message.images[0].image_url.url (data URL).
    """
    t0 = time.perf_counter()
    attempt = 0
    last_err: str = ""

    # Normalize modalities
    mods = [str(m) for m in (modalities or []) if str(m).strip()]
    if not mods:
        mods = ["image", "text"]

    # Build a data URL for the input image
    img_data_url = f"data:{src_mime};base64,{src_b64}"

    prompt_head = str(prompt or "").replace("\n", " ").strip()
    prompt_head = _truncate_text(prompt_head, 200)
    debug_ctx = {
        "provider": "openrouter",
        "base_url": str(base_url),
        "model": str(model),
        "modalities": list(mods),
        "image_config": dict(image_cfg) if isinstance(image_cfg, dict) else None,
        "src_mime": str(src_mime),
        "src_b64_len": int(len(src_b64 or "")),
        "prompt_len": int(len(prompt or "")),
        "prompt_head": prompt_head,
    }

    while attempt < int(max_retries):
        attempt += 1
        try:
            payload: Dict[str, Any] = {
                "model": str(model),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_data_url}},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                "modalities": list(mods),
                "stream": False,
            }
            if isinstance(image_cfg, dict) and image_cfg:
                # OpenRouter uses `image_config` (snake_case keys like aspect_ratio/image_size).
                payload["image_config"] = dict(image_cfg)

            resp_json = _http_post_json(
                str(base_url), headers=headers, payload=payload, timeout_s=int(timeout_s)
            )
            img_bytes, err = _extract_first_image_bytes_openrouter(resp_json)
            if err:
                raise RuntimeError(err)
            assert img_bytes is not None

            meta = _postprocess_and_write(
                img_bytes,
                out_path=out_path,
                output_ext=output_ext,
                post_cfg=post_cfg,
            )
            return {
                "status": "success",
                "attempts": int(attempt),
                "elapsed_s": float(time.perf_counter() - t0),
                **dict(meta or {}),
                "error": "",
            }
        except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
            last_err = str(e)
            if attempt >= int(max_retries):
                return {
                    "status": "failed",
                    "attempts": int(attempt),
                    "elapsed_s": float(time.perf_counter() - t0),
                    "orig_w": 0,
                    "orig_h": 0,
                    "aspect_w": 0,
                    "aspect_h": 0,
                    "out_w": 0,
                    "out_h": 0,
                    "pad_area_px": 0,
                    "area_scale": 1.0,
                    "error": last_err + "\nDEBUG " + _truncate_text(json.dumps(debug_ctx, ensure_ascii=False), 2000),
                }
            time.sleep(float(retry_backoff_s) * float(attempt))
        except Exception as e:
            last_err = str(e)
            return {
                "status": "failed",
                "attempts": int(attempt),
                "elapsed_s": float(time.perf_counter() - t0),
                "orig_w": 0,
                "orig_h": 0,
                "aspect_w": 0,
                "aspect_h": 0,
                "out_w": 0,
                "out_h": 0,
                "pad_area_px": 0,
                "area_scale": 1.0,
                "error": last_err + "\nDEBUG " + _truncate_text(json.dumps(debug_ctx, ensure_ascii=False), 2000),
            }
    return {
        "status": "failed",
        "attempts": int(attempt),
        "elapsed_s": float(time.perf_counter() - t0),
        "orig_w": 0,
        "orig_h": 0,
        "aspect_w": 0,
        "aspect_h": 0,
        "out_w": 0,
        "out_h": 0,
        "pad_area_px": 0,
        "area_scale": 1.0,
        "error": last_err or "unknown error",
    }


@dataclass(frozen=True)
class AugSpec:
    name: str
    prompt: str
    num_images: int = 1


def _load_aug_specs(cfg: Dict[str, Any]) -> Tuple[str, List[AugSpec]]:
    prefix = str(cfg.get("prompt_prefix", "") or "")
    aug_list = cfg.get("augmentations", []) or []
    if not isinstance(aug_list, list) or not aug_list:
        raise ValueError("Config must include a non-empty `augmentations:` list.")
    out: List[AugSpec] = []
    for i, item in enumerate(aug_list):
        if not isinstance(item, dict):
            raise ValueError(f"augmentations[{i}] must be a mapping")
        name = str(item.get("name", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not name:
            raise ValueError(f"augmentations[{i}] missing non-empty `name`")
        if not prompt:
            raise ValueError(f"augmentations[{i}] missing non-empty `prompt`")
        try:
            num_images = int(item.get("num_images", 1))
        except Exception:
            num_images = 1
        num_images = max(1, num_images)
        out.append(AugSpec(name=name, prompt=prompt, num_images=num_images))
    return prefix, out


def _load_train_image_index(
    *,
    data_root: Path,
    train_csv: str,
    primary_targets: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Returns a dataframe with columns: [image_id, image_path]
    - image_path is the CSV path (relative to data_root in this repo)
    """
    csv_path = data_root / train_csv
    if not csv_path.is_file():
        raise FileNotFoundError(f"train_csv not found: {csv_path}")
    df = pd.read_csv(str(csv_path))
    if "sample_id" not in df.columns or "image_path" not in df.columns:
        raise ValueError("train.csv must contain columns: sample_id, image_path")
    df = df.copy()
    df["image_id"] = df["sample_id"].astype(str).str.split("__", n=1, expand=True)[0]
    if primary_targets:
        # Keep only image_ids that have all required primary targets present (non-null).
        need = set([str(t) for t in primary_targets])
        if "target_name" in df.columns and "target" in df.columns:
            ok = (
                df[df["target_name"].astype(str).isin(list(need))]
                .dropna(subset=["target"])
                .groupby("image_id")["target_name"]
                .agg(lambda s: set(map(str, s.tolist())))
            )
            good_ids = set([idx for idx, seen in ok.items() if need.issubset(seen)])
            df = df[df["image_id"].isin(list(good_ids))]
        # else: if the columns are missing, we can't filter; proceed with all images.
    img = df.groupby("image_id")["image_path"].first().reset_index()
    img = img.sort_values("image_id").reset_index(drop=True)
    return img


def _load_done_set(manifest_path: Path) -> set[Tuple[str, str, int]]:
    """
    Completed keys: (image_id, aug_name, variant_idx)
    """
    done: set[Tuple[str, str, int]] = set()
    if not manifest_path.is_file():
        return done
    try:
        df = pd.read_csv(str(manifest_path))
    except Exception:
        return done
    if "status" in df.columns:
        df = df[df["status"].astype(str) == "success"]
    for _i, row in df.iterrows():
        try:
            image_id = str(row.get("image_id", "")).strip()
            aug_name = str(row.get("aug_name", "")).strip()
            variant_idx = int(row.get("variant_idx", 0))
        except Exception:
            continue
        if image_id and aug_name:
            done.add((image_id, aug_name, variant_idx))
    return done


MANIFEST_COLUMNS_V2: List[str] = [
    "image_id",
    "aug_name",
    "variant_idx",
    "src_image_path",
    "image_path",
    "status",
    "error",
    "ts",
    # --- postprocess / padding metadata (optional, used for label correction) ---
    # original (model) output size before enforcing aspect/resize
    "orig_w",
    "orig_h",
    # size after enforcing aspect ratio (pad/crop) but before final resize
    "aspect_w",
    "aspect_h",
    # final written image size
    "out_w",
    "out_h",
    # padding area in pixels (>=0; for crop mode this stays 0)
    "pad_area_px",
    # area_scale = (aspect_w*aspect_h)/(orig_w*orig_h)  (>=1 for pad, <=1 for crop)
    "area_scale",
]


def _ensure_manifest_schema(manifest_path: Path, required_cols: Sequence[str]) -> None:
    """
    Ensure manifest exists and has a header compatible with `required_cols`.
    If an existing manifest has fewer columns, we rewrite it with the new schema
    and fill missing columns with defaults.
    """
    required = [str(c) for c in required_cols]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(required)
        return

    # Read header only
    try:
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r, [])
    except Exception:
        header = []

    header = [str(h).strip() for h in header if str(h).strip()]
    if not header:
        # Corrupt/empty header: rewrite
        tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(required)
        os.replace(str(tmp), str(manifest_path))
        return

    # Already compatible if it contains all required columns (order doesn't matter for pandas)
    if all(c in header for c in required):
        return

    # Migrate: read existing CSV and rewrite with required schema
    try:
        df = pd.read_csv(str(manifest_path))
    except Exception:
        # If parsing fails, keep the old manifest as backup and start a new one
        bak = manifest_path.with_suffix(manifest_path.suffix + ".bak")
        try:
            os.replace(str(manifest_path), str(bak))
        except Exception:
            pass
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(required)
        return

    # Fill missing columns with defaults
    for c in required:
        if c in df.columns:
            continue
        # Reasonable defaults
        if c in ("orig_w", "orig_h", "aspect_w", "aspect_h", "out_w", "out_h", "pad_area_px"):
            df[c] = 0
        elif c == "area_scale":
            df[c] = 1.0
        else:
            df[c] = ""

    # Reorder columns to required schema
    df = df[list(required)]
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    df.to_csv(str(tmp), index=False)
    os.replace(str(tmp), str(manifest_path))


def _append_manifest_row(manifest_path: Path, row: Sequence[Any]) -> None:
    with open(manifest_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(row))


def _resolve_path_best_effort(repo_root: Path, path_str: str) -> Path:
    p = Path(str(path_str)).expanduser()
    if p.is_absolute():
        return p
    return (repo_root / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(resolve_repo_root() / "configs" / "nano_banana_pro_augment.yaml"),
        help="Path to nano banana pro augmentation YAML config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of images to process (0 => no limit).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate even if output already exists (still writes to manifest).",
    )
    args = parser.parse_args()

    repo_root = resolve_repo_root()
    cfg_path = _resolve_path_best_effort(repo_root, str(args.config))
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping at the top level.")

    api_cfg = dict(cfg.get("api", {}) or {})
    data_cfg = dict(cfg.get("data", {}) or {})

    provider = str(api_cfg.get("provider", "gemini")).strip().lower()
    if provider not in ("gemini", "openrouter"):
        raise ValueError(f"Unsupported api.provider: {provider} (expected: gemini|openrouter)")

    default_key_file = "openrouter.secret" if provider == "openrouter" else "gemini.secret"
    api_key_file = str(api_cfg.get("api_key_file", default_key_file))
    api_key_path = _resolve_path_best_effort(repo_root, api_key_file)
    api_key = _read_text_file(api_key_path)
    if not api_key:
        raise ValueError(f"Empty API key file: {api_key_path}")

    model = str(api_cfg.get("model", "gemini-3-pro-image-preview")).strip()
    timeout_s = int(api_cfg.get("timeout_s", 120))
    max_retries = int(api_cfg.get("max_retries", 3))
    retry_backoff_s = float(api_cfg.get("retry_backoff_s", 5.0))
    try:
        concurrency = int(api_cfg.get("concurrency", 1))
    except Exception:
        concurrency = 1
    concurrency = max(1, int(concurrency))

    # Concurrency scope:
    # - per_image: (default) run variants of a single image in parallel, then move to next image.
    # - global: keep up to `concurrency` in-flight jobs across the whole dataset.
    concurrency_scope = (
        str(api_cfg.get("concurrency_scope", "per_image")).strip().lower().replace("-", "_")
    )
    if concurrency_scope in ("perimage", "image"):
        concurrency_scope = "per_image"
    if concurrency_scope not in ("per_image", "global"):
        raise ValueError(
            f"Unsupported api.concurrency_scope: {concurrency_scope} (expected: per_image|global)"
        )

    data_root = _resolve_path_best_effort(repo_root, str(data_cfg.get("data_root", "data")))
    train_csv = str(data_cfg.get("train_csv", "train.csv"))
    output_dir = _resolve_path_best_effort(
        repo_root, str(data_cfg.get("output_dir", "data/nano_banana_pro/train"))
    )
    output_ext = str(data_cfg.get("output_ext", "png")).lstrip(".").strip() or "png"
    manifest_name = str(data_cfg.get("manifest", "manifest.csv"))
    manifest_path = output_dir / manifest_name

    primary_targets = data_cfg.get("primary_targets", ["Dry_Total_g"])
    if primary_targets is None:
        primary_targets_list = None
    elif isinstance(primary_targets, (list, tuple)):
        primary_targets_list = [str(x) for x in primary_targets]
    else:
        primary_targets_list = [str(primary_targets)]

    prompt_prefix, aug_specs = _load_aug_specs(cfg)
    post_cfg = dict(cfg.get("postprocess", {}) or {})

    # Init output dir + manifest + log file
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / str(data_cfg.get("log_file", "augment.log"))
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    logger.add(str(log_path), level="INFO", enqueue=True)

    _ensure_manifest_schema(manifest_path, MANIFEST_COLUMNS_V2)
    done = _load_done_set(manifest_path)

    # Build image index
    img_df = _load_train_image_index(
        data_root=data_root, train_csv=train_csv, primary_targets=primary_targets_list
    )
    if args.limit and int(args.limit) > 0:
        img_df = img_df.head(int(args.limit)).reset_index(drop=True)

    logger.info("Config: {}", cfg_path)
    logger.info("Data root: {}", data_root)
    logger.info("Train CSV: {}", data_root / train_csv)
    logger.info("Output dir: {}", output_dir)
    logger.info("Manifest: {}", manifest_path)
    logger.info("API provider: {}", provider)
    logger.info("Model: {}", model)
    logger.info("Total images: {}", len(img_df))
    logger.info("Augmentations: {}", [a.name for a in aug_specs])
    logger.info("Concurrency: {} (parallel API calls)", concurrency)
    logger.info("Concurrency scope: {}", concurrency_scope)

    num_variants_total = int(sum(int(a.num_images) for a in aug_specs))
    total_jobs_est = int(len(img_df) * num_variants_total) if len(img_df) > 0 else 0
    logger.info("Planned jobs: {} ({} images * {} variants)", total_jobs_est, len(img_df), num_variants_total)
    logger.info("Starting augmentation run...")

    if provider == "openrouter":
        base_url = str(api_cfg.get("base_url", "https://openrouter.ai/api/v1/chat/completions")).strip()
        headers: Dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        http_referer = str(api_cfg.get("http_referer", "") or "").strip()
        x_title = str(api_cfg.get("x_title", "") or "").strip()
        if http_referer:
            headers["HTTP-Referer"] = http_referer
        if x_title:
            headers["X-Title"] = x_title
        extra_headers = api_cfg.get("extra_headers", None)
        if isinstance(extra_headers, dict):
            for k, v in extra_headers.items():
                ks = str(k).strip()
                if not ks:
                    continue
                headers[ks] = str(v)
    else:
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        headers = {"x-goog-api-key": api_key}

    # OpenRouter-specific optional controls
    modalities_cfg = api_cfg.get("modalities", ["image", "text"])
    if modalities_cfg is None:
        modalities = ["image", "text"]
    elif isinstance(modalities_cfg, (list, tuple)):
        modalities = [str(x) for x in modalities_cfg]
    else:
        modalities = [str(modalities_cfg)]

    processed = 0
    skipped = 0
    failed = 0
    missing_src = 0
    jobs_done = 0
    jobs_started = 0
    t0_all = time.perf_counter()

    # Global-concurrency bookkeeping (used only when concurrency_scope == "global")
    futures_global: Dict[Any, Dict[str, Any]] = {}
    img_states: Dict[str, Dict[str, Any]] = {}

    def _maybe_log_image_done(image_id_done: str) -> None:
        """
        For global concurrency: when all variants (including skips) for an image are accounted for,
        emit the same per-image summary log as per-image mode.
        """
        st = img_states.get(str(image_id_done))
        if not st:
            return
        try:
            done_n = int(st.get("done", 0))
            total_n = int(st.get("total", 0))
        except Exception:
            return
        if total_n <= 0 or done_n < total_n:
            return
        dt_img = time.perf_counter() - float(st.get("t0", t0_all))
        dt_all = time.perf_counter() - t0_all
        logger.info(
            "Image done: image_id={} ok={} skip={} fail={} time={} | overall jobs {}/{} ({:.1f}%) elapsed={} (processed={} skipped={} failed={} missing_src={})",
            str(image_id_done),
            int(st.get("ok", 0)),
            int(st.get("skip", 0)),
            int(st.get("fail", 0)),
            _format_seconds(dt_img),
            jobs_done,
            total_jobs_est,
            _pct(jobs_done, total_jobs_est),
            _format_seconds(dt_all),
            processed,
            skipped,
            failed,
            missing_src,
        )
        # Free memory
        img_states.pop(str(image_id_done), None)

    def _consume_completed_future(fut: Any, meta: Dict[str, Any]) -> Tuple[str, str]:
        """
        Consume a finished job future:
        - writes manifest row
        - updates global counters
        - logs OK/FAIL
        Returns (image_id, status).
        """
        nonlocal processed, failed, jobs_done

        image_id_job = str(meta.get("image_id", "") or "").strip()
        src_image_path_job = str(meta.get("src_image_path", "") or "").strip()
        aug_name = str(meta.get("aug_name", ""))
        variant_idx = int(meta.get("variant_idx", 0))
        out_name = str(meta.get("out_name", ""))
        out_path = meta.get("out_path", None)
        ts = int(meta.get("ts", int(time.time())))
        key = meta.get("key", None)

        try:
            res = fut.result()
        except Exception as e:
            res = {
                "status": "failed",
                "attempts": 1,
                "elapsed_s": 0.0,
                "orig_w": 0,
                "orig_h": 0,
                "aspect_w": 0,
                "aspect_h": 0,
                "out_w": 0,
                "out_h": 0,
                "pad_area_px": 0,
                "area_scale": 1.0,
                "error": str(e),
            }

        jobs_done += 1
        dt_job = float(res.get("elapsed_s", 0.0))
        attempts = int(res.get("attempts", 1))
        status = str(res.get("status", "failed"))

        # Store manifest image_path relative to data_root when possible
        try:
            rel_out = (
                os.path.relpath(str(out_path), str(data_root)) if out_path is not None else out_name
            )
        except Exception:
            rel_out = str(out_path) if out_path is not None else out_name

        # Padding / area metadata (used for per-sample area correction during training)
        orig_w = int(res.get("orig_w", 0))
        orig_h = int(res.get("orig_h", 0))
        aspect_w = int(res.get("aspect_w", 0))
        aspect_h = int(res.get("aspect_h", 0))
        out_w = int(res.get("out_w", 0))
        out_h = int(res.get("out_h", 0))
        pad_area_px = int(res.get("pad_area_px", 0))
        try:
            area_scale = float(res.get("area_scale", 1.0))
        except Exception:
            area_scale = 1.0
        if not (area_scale > 0.0):
            area_scale = 1.0

        if status == "success":
            processed += 1
            if isinstance(key, tuple):
                done.add(key)
            _append_manifest_row(
                manifest_path,
                [
                    image_id_job,
                    aug_name,
                    int(variant_idx),
                    src_image_path_job,
                    str(rel_out).replace("\\", "/"),
                    "success",
                    "",
                    ts,
                    orig_w,
                    orig_h,
                    aspect_w,
                    aspect_h,
                    out_w,
                    out_h,
                    pad_area_px,
                    area_scale,
                ],
            )
            size_str = f"{out_w}x{out_h}" if (out_w > 0 and out_h > 0) else "unknown"
            logger.info(
                "  [{}/{} {:.1f}%] OK   aug={} v={} attempts={} time={} size={} -> {}",
                jobs_done,
                total_jobs_est,
                _pct(jobs_done, total_jobs_est),
                aug_name,
                variant_idx,
                attempts,
                _format_seconds(dt_job),
                size_str,
                out_name,
            )
            return image_id_job, "success"

        failed += 1
        err_full = str(res.get("error", "") or "")
        err = err_full[:500]
        err_log = _truncate_text(err_full, 4000)
        _append_manifest_row(
            manifest_path,
            [
                image_id_job,
                aug_name,
                int(variant_idx),
                src_image_path_job,
                str(rel_out).replace("\\", "/"),
                "failed",
                err,
                ts,
                orig_w,
                orig_h,
                aspect_w,
                aspect_h,
                out_w,
                out_h,
                pad_area_px,
                area_scale,
            ],
        )
        logger.error(
            "  [{}/{} {:.1f}%] FAIL aug={} v={} attempts={} time={} err={}",
            jobs_done,
            total_jobs_est,
            _pct(jobs_done, total_jobs_est),
            aug_name,
            variant_idx,
            attempts,
            _format_seconds(dt_job),
            err_log,
        )
        return image_id_job, "failed"

    def _drain_one_global() -> None:
        """
        Wait for at least one in-flight future to complete, then consume it.
        Used to keep max in-flight futures <= `concurrency` when concurrency_scope == "global".
        """
        if not futures_global:
            return
        done_set, _ = wait(list(futures_global.keys()), return_when=FIRST_COMPLETED)
        for f in done_set:
            meta = futures_global.pop(f, {})
            image_id_done, status_done = _consume_completed_future(f, meta)
            st = img_states.get(str(image_id_done))
            if st:
                st["done"] = int(st.get("done", 0)) + 1
                if status_done == "success":
                    st["ok"] = int(st.get("ok", 0)) + 1
                else:
                    st["fail"] = int(st.get("fail", 0)) + 1
                _maybe_log_image_done(str(image_id_done))

    with ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        for idx, row in img_df.iterrows():
            t0_img = time.perf_counter()
            image_id = str(row["image_id"]).strip()
            rel_image_path = str(row["image_path"]).strip()
            if not image_id or not rel_image_path:
                continue
            src_path = (data_root / rel_image_path).resolve()
            if not src_path.is_file():
                missing_src += 1
                # Advance progress by the number of variants we would have generated for this image
                jobs_done += int(num_variants_total)
                logger.warning(
                    "[{}/{} {:.1f}%] MISSING src image_id={} path={}",
                    jobs_done,
                    total_jobs_est,
                    _pct(jobs_done, total_jobs_est),
                    image_id,
                    src_path,
                )
                continue

            logger.info(
                "Image {}/{}: image_id={} src={} (variants={})",
                idx + 1,
                len(img_df),
                image_id,
                rel_image_path,
                num_variants_total,
            )

            src_bytes = src_path.read_bytes()
            src_b64 = base64.b64encode(src_bytes).decode("utf-8")
            src_mime = _guess_mime_type(src_path)

            img_ok = 0
            img_skip = 0
            img_fail = 0

            # Submit jobs for this image.
            # - per_image: futures are per-image and we wait for them before moving on.
            # - global: futures are shared and we keep up to `concurrency` in-flight across the dataset.
            if concurrency_scope == "global":
                img_states[image_id] = {
                    "t0": t0_img,
                    "total": int(num_variants_total),
                    "done": 0,
                    "ok": 0,
                    "skip": 0,
                    "fail": 0,
                }
                futures = futures_global
            else:
                futures: Dict[Any, Dict[str, Any]] = {}
            for aug in aug_specs:
                for variant_idx in range(int(aug.num_images)):
                    key = (image_id, aug.name, int(variant_idx))
                    out_name = f"{image_id}__{aug.name}__v{variant_idx}.{output_ext}"
                    out_path = output_dir / out_name

                    if not args.force:
                        if key in done and out_path.is_file() and out_path.stat().st_size > 0:
                            skipped += 1
                            img_skip += 1
                            jobs_done += 1
                            logger.info(
                                "  [{}/{} {:.1f}%] SKIP aug={} v={} (manifest+file) -> {}",
                                jobs_done,
                                total_jobs_est,
                                _pct(jobs_done, total_jobs_est),
                                aug.name,
                                variant_idx,
                                out_name,
                            )
                            if concurrency_scope == "global":
                                st = img_states.get(image_id)
                                if st:
                                    st["skip"] = int(st.get("skip", 0)) + 1
                                    st["done"] = int(st.get("done", 0)) + 1
                                    _maybe_log_image_done(image_id)
                            continue
                        if out_path.is_file() and out_path.stat().st_size > 0:
                            # Even if not in manifest (older runs), treat as done.
                            skipped += 1
                            img_skip += 1
                            jobs_done += 1
                            logger.info(
                                "  [{}/{} {:.1f}%] SKIP aug={} v={} (file exists) -> {}",
                                jobs_done,
                                total_jobs_est,
                                _pct(jobs_done, total_jobs_est),
                                aug.name,
                                variant_idx,
                                out_name,
                            )
                            if concurrency_scope == "global":
                                st = img_states.get(image_id)
                                if st:
                                    st["skip"] = int(st.get("skip", 0)) + 1
                                    st["done"] = int(st.get("done", 0)) + 1
                                    _maybe_log_image_done(image_id)
                            continue

                    prompt = aug.prompt.strip()
                    if prompt_prefix.strip():
                        prompt = prompt_prefix.strip() + "\n\n" + prompt
                    # Optional imageConfig passthrough
                    image_cfg = api_cfg.get("image_config", None)

                    # Log at job start + submit to thread pool
                    ts = int(time.time())
                    jobs_started += 1
                    in_flight = len(futures) + 1
                    logger.info(
                        "  [{}/{} {:.1f}%] START aug={} v={} -> {} (in_flight={}/{})",
                        jobs_done,
                        total_jobs_est,
                        _pct(jobs_done, total_jobs_est),
                        aug.name,
                        variant_idx,
                        out_name,
                        in_flight,
                        concurrency,
                    )

                    if concurrency_scope == "global":
                        # Keep in-flight requests bounded to `concurrency`.
                        while len(futures_global) >= int(concurrency):
                            _drain_one_global()

                    if provider == "openrouter":
                        fut = pool.submit(
                            _run_openrouter_job,
                            base_url=base_url,
                            headers=headers,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                            model=model,
                            modalities=modalities,
                            src_mime=src_mime,
                            src_b64=src_b64,
                            prompt=prompt,
                            image_cfg=dict(image_cfg) if isinstance(image_cfg, dict) else None,
                            output_ext=output_ext,
                            post_cfg=post_cfg,
                            out_path=out_path,
                        )
                    else:
                        fut = pool.submit(
                            _run_gemini_job,
                            base_url=base_url,
                            headers=headers,
                            timeout_s=timeout_s,
                            max_retries=max_retries,
                            retry_backoff_s=retry_backoff_s,
                            src_mime=src_mime,
                            src_b64=src_b64,
                            prompt=prompt,
                            image_cfg=dict(image_cfg) if isinstance(image_cfg, dict) else None,
                            output_ext=output_ext,
                            post_cfg=post_cfg,
                            out_path=out_path,
                        )
                    futures[fut] = {
                        "image_id": image_id,
                        "aug_name": aug.name,
                        "variant_idx": int(variant_idx),
                        "src_image_path": rel_image_path,
                        "out_name": out_name,
                        "out_path": out_path,
                        "key": key,
                        "ts": ts,
                    }

            if concurrency_scope != "global":
                # Collect results for this image
                for fut in as_completed(list(futures.keys())):
                    meta = futures.get(fut, {})
                    _image_id_done, status_done = _consume_completed_future(fut, meta)
                    if status_done == "success":
                        img_ok += 1
                    else:
                        img_fail += 1

                dt_img = time.perf_counter() - t0_img
                dt_all = time.perf_counter() - t0_all
                logger.info(
                    "Image done: image_id={} ok={} skip={} fail={} time={} | overall jobs {}/{} ({:.1f}%) elapsed={} (processed={} skipped={} failed={} missing_src={})",
                    image_id,
                    img_ok,
                    img_skip,
                    img_fail,
                    _format_seconds(dt_img),
                    jobs_done,
                    total_jobs_est,
                    _pct(jobs_done, total_jobs_est),
                    _format_seconds(dt_all),
                    processed,
                    skipped,
                    failed,
                    missing_src,
                )

        if concurrency_scope == "global":
            # Drain remaining in-flight jobs
            for fut in as_completed(list(futures_global.keys())):
                meta = futures_global.pop(fut, {})
                image_id_done, status_done = _consume_completed_future(fut, meta)
                st = img_states.get(str(image_id_done))
                if st:
                    st["done"] = int(st.get("done", 0)) + 1
                    if status_done == "success":
                        st["ok"] = int(st.get("ok", 0)) + 1
                    else:
                        st["fail"] = int(st.get("fail", 0)) + 1
                    _maybe_log_image_done(str(image_id_done))

            # Best-effort: if any image states remain, log them for debugging.
            if img_states:
                logger.warning(
                    "[GLOBAL] Some images did not reach done==total; this may indicate an internal bookkeeping bug. remaining={}",
                    list(img_states.keys())[:20],
                )

    logger.info(
        "Done. processed={} skipped={} failed={} missing_src={} jobs_done={}/{} elapsed={} manifest={}",
        processed,
        skipped,
        failed,
        missing_src,
        jobs_done,
        total_jobs_est,
        _format_seconds(time.perf_counter() - t0_all),
        manifest_path,
    )


if __name__ == "__main__":
    main()


