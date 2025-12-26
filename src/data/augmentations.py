import math
import os
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageFilter


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.01, p: float = 0.5) -> None:
        self.mean = float(mean)
        self.std = float(std)
        self.p = float(p)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return tensor
        if torch.rand(()) > self.p:
            return tensor
        if not torch.is_tensor(tensor):
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        out = tensor + noise
        return out.clamp(0.0, 1.0)


class ProbGaussianBlur:
    def __init__(
        self,
        p: float = 0.5,
        kernel_size_range: Tuple[int, int] = (3, 5),
        sigma: Union[Tuple[float, float], float] = (0.1, 1.0),
    ) -> None:
        self.p = float(p)
        self.kernel_size_range = (int(kernel_size_range[0]), int(kernel_size_range[1]))
        self.sigma = sigma

    @staticmethod
    def _sample_odd_in_range(lo: int, hi: int) -> int:
        lo = max(3, int(lo))
        hi = max(lo, int(hi))
        k = random.randint(lo, hi)
        if k % 2 == 0:
            k = k + 1 if k + 1 <= hi else k - 1
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        return k

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0:
            return tensor
        if torch.rand(()) > self.p:
            return tensor
        if not torch.is_tensor(tensor):
            return tensor
        k = self._sample_odd_in_range(*self.kernel_size_range)
        if isinstance(self.sigma, (list, tuple)):
            sig = float(random.uniform(float(self.sigma[0]), float(self.sigma[1])))
        else:
            sig = float(self.sigma)
        try:
            return F.gaussian_blur(tensor, kernel_size=[k, k], sigma=[sig, sig])
        except Exception:
            # Fallback to transform module if functional API not available
            gb = T.GaussianBlur(kernel_size=k, sigma=sig)
            return gb(tensor)


class RandomWatermark:
    def __init__(
        self,
        p: float = 0.3,
        texts: Optional[List[str]] = None,
        timestamp_prob: float = 0.5,
        timestamp_format: str = "%Y-%m-%d %H:%M:%S",
        font_path: Optional[str] = None,
        font_size_frac_range: Tuple[float, float] = (0.04, 0.12),
        alpha_range: Tuple[int, int] = (128, 200),
        color_choices: Optional[List[Tuple[int, int, int]]] = None,
        color_jitter: int = 0,
        use_random_text: bool = False,
        random_text_length_range: Tuple[int, int] = (6, 12),
        charset: Optional[str] = None,
        # New: camera-like styling controls (all optional; defaults keep legacy behavior).
        style: str = "random",  # "random" | "camera_timestamp"
        position: str = "random",  # "random" | "bottom_right" | "bottom_left" | "top_left" | "top_right"
        position_choices: Optional[List[str]] = None,
        position_probs: Optional[List[float]] = None,
        margin_frac_range: Tuple[float, float] = (0.01, 0.04),
        render_mode: Optional[str] = None,  # None => style-dependent default ("stroke" for random, "shadow" for camera)
        # Shadow rendering (used when render_mode == "shadow")
        shadow_color: Tuple[int, int, int] = (0, 0, 0),
        shadow_alpha_range: Tuple[int, int] = (140, 255),
        shadow_offset_frac_range: Tuple[float, float] = (0.03, 0.07),  # of font_size
        # Stroke rendering (used when render_mode == "stroke")
        stroke_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_alpha_mult: float = 1.0,
        stroke_width_frac: float = 1.0 / 15.0,
    ) -> None:
        self.p = float(p)
        self.texts = list(texts or ["CSIRO", "Biomass", "Train", "Sample"])
        self.timestamp_prob = float(timestamp_prob)
        self.timestamp_format = str(timestamp_format or "%Y-%m-%d %H:%M:%S")
        self.font_path = font_path
        self.font_size_frac_range = (float(font_size_frac_range[0]), float(font_size_frac_range[1]))
        self.alpha_range = (int(alpha_range[0]), int(alpha_range[1]))
        self.color_choices = list(color_choices or [(255, 255, 255), (255, 230, 0), (0, 255, 255), (255, 128, 0)])
        self.color_jitter = int(color_jitter)
        self.use_random_text = bool(use_random_text)
        self.random_text_length_range = (int(random_text_length_range[0]), int(random_text_length_range[1]))
        default_charset = string.ascii_uppercase + string.digits
        self.charset = str(charset) if (charset and isinstance(charset, str) and len(charset) > 0) else default_charset

        self.style = str(style or "random").strip().lower()
        self.position = str(position or "random").strip().lower()
        self.position_choices: Optional[List[str]] = None
        self.position_probs: Optional[List[float]] = None
        if position_choices is not None:
            try:
                choices = [str(x).strip().lower() for x in list(position_choices) if str(x).strip() != ""]
            except Exception:
                choices = []
            if len(choices) > 0:
                self.position_choices = choices
                if position_probs is not None:
                    try:
                        probs = [float(x) for x in list(position_probs)]
                    except Exception:
                        probs = []
                    if len(probs) == len(choices) and any(p > 0.0 for p in probs):
                        self.position_probs = probs
        self.margin_frac_range = (float(margin_frac_range[0]), float(margin_frac_range[1]))

        # Defaults by style: keep legacy look for "random"; use shadow for camera-like timestamps.
        if render_mode is None:
            self.render_mode = "shadow" if self.style in {"camera", "camera_timestamp"} else "stroke"
        else:
            self.render_mode = str(render_mode).strip().lower()

        self.shadow_color = (int(shadow_color[0]), int(shadow_color[1]), int(shadow_color[2]))
        self.shadow_alpha_range = (int(shadow_alpha_range[0]), int(shadow_alpha_range[1]))
        self.shadow_offset_frac_range = (float(shadow_offset_frac_range[0]), float(shadow_offset_frac_range[1]))

        self.stroke_color = (int(stroke_color[0]), int(stroke_color[1]), int(stroke_color[2]))
        self.stroke_alpha_mult = float(stroke_alpha_mult)
        self.stroke_width_frac = float(stroke_width_frac)

    @staticmethod
    def _iter_default_font_paths() -> List[str]:
        # Prefer bold, scalable system fonts. Works well on most Linux distros.
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        ]
        return [p for p in candidates if os.path.exists(p)]

    def _choose_font(self, size: int) -> ImageFont.ImageFont:
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size=size)
        except Exception:
            pass
        try:
            # Try common system fonts (scalable). This fixes "tiny watermark" when no font_path is set.
            for p in self._iter_default_font_paths():
                try:
                    return ImageFont.truetype(p, size=size)
                except Exception:
                    continue
            # Try by name (Pillow may resolve these via fontconfig on some setups)
            for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Regular.ttf"]:
                try:
                    return ImageFont.truetype(name, size=size)
                except Exception:
                    continue
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

    def _sample_timestamp_text(self) -> str:
        # Generate a reproducible pseudo-timestamp based only on RNG state,
        # which is seeded globally via pl.seed_everything(cfg['seed'], workers=True).
        base = datetime(2010, 1, 1, 0, 0, 0)
        # Sample up to ~20 years worth of seconds for variety.
        max_offset_seconds = 20 * 365 * 24 * 60 * 60
        offset_seconds = random.randint(0, max_offset_seconds)
        ts = base + timedelta(seconds=offset_seconds)
        fmt = self.timestamp_format or "%Y-%m-%d %H:%M:%S"
        try:
            return ts.strftime(fmt)
        except Exception:
            return ts.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _clamp_int(x: int, lo: int, hi: int) -> int:
        return lo if x < lo else (hi if x > hi else int(x))

    @staticmethod
    def _sample_int_range(lo: int, hi: int) -> int:
        lo_i, hi_i = int(lo), int(hi)
        if hi_i < lo_i:
            hi_i = lo_i
        return random.randint(lo_i, hi_i)

    def _sample_margin_px(self, *, w: int, h: int) -> Tuple[int, int]:
        lo_f, hi_f = float(self.margin_frac_range[0]), float(self.margin_frac_range[1])
        lo_f, hi_f = (lo_f, hi_f) if hi_f >= lo_f else (hi_f, lo_f)
        base = max(0.0, random.uniform(lo_f, hi_f))
        m = int(round(min(w, h) * base))
        return max(0, m), max(0, m)

    def _pick_position_bbox_topleft(self, *, w: int, h: int, tw: int, th: int, position: str) -> Tuple[int, int]:
        # Returns the top-left corner for the *text bounding box* (not the draw origin).
        pos = str(position or "random").strip().lower()
        if pos in {"bottom_right", "br"}:
            mx, my = self._sample_margin_px(w=w, h=h)
            return max(0, w - tw - mx), max(0, h - th - my)
        if pos in {"bottom_left", "bl"}:
            mx, my = self._sample_margin_px(w=w, h=h)
            return mx, max(0, h - th - my)
        if pos in {"top_left", "tl"}:
            mx, my = self._sample_margin_px(w=w, h=h)
            return mx, my
        if pos in {"top_right", "tr"}:
            mx, my = self._sample_margin_px(w=w, h=h)
            return max(0, w - tw - mx), my
        # Legacy random placement
        x0 = random.randint(0, max(0, w - tw)) if w > tw else 0
        y0 = random.randint(0, max(0, h - th)) if h > th else 0
        return x0, y0

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0:
            return img
        if torch.rand(()) > self.p:
            return img
        if not isinstance(img, Image.Image):
            return img
        w, h = img.size
        rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Text selection: by style, we can force a camera-like timestamp.
        if self.style in {"camera", "camera_timestamp"}:
            text = self._sample_timestamp_text()
        else:
            use_ts = random.random() < self.timestamp_prob
            if use_ts:
                text = self._sample_timestamp_text()
            else:
                if self.use_random_text:
                    lo, hi = self.random_text_length_range
                    if hi < lo:
                        hi = lo
                    length = random.randint(max(1, lo), max(1, hi))
                    text = "".join(random.choices(self.charset, k=length))
                else:
                    # fallback to provided texts
                    choices = self.texts if len(self.texts) > 0 else ["WATERMARK"]
                    text = random.choice(choices)

        font_size = max(12, int(min(w, h) * random.uniform(*self.font_size_frac_range)))
        font = self._choose_font(font_size)
        color = random.choice(self.color_choices)
        if self.color_jitter:
            j = abs(int(self.color_jitter))
            if j > 0:
                r = self._clamp_int(int(color[0]) + random.randint(-j, j), 0, 255)
                g = self._clamp_int(int(color[1]) + random.randint(-j, j), 0, 255)
                b = self._clamp_int(int(color[2]) + random.randint(-j, j), 0, 255)
                color = (r, g, b)
        alpha = int(random.randint(*self.alpha_range))
        fill = (int(color[0]), int(color[1]), int(color[2]), alpha)

        # Positioning ensuring the text fits. We position using the text bbox (handles fonts with negative offsets).
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            left, top, right, bottom = bbox
            tw, th = right - left, bottom - top
        except Exception:
            left, top, tw, th = 0, 0, font_size * len(text), int(font_size * 1.2)

        pos = self.position
        if self.position_choices is not None:
            try:
                pos = random.choices(self.position_choices, weights=self.position_probs, k=1)[0]
            except Exception:
                pos = random.choice(self.position_choices)
        x0, y0 = self._pick_position_bbox_topleft(w=w, h=h, tw=int(tw), th=int(th), position=pos)
        # draw origin so that bbox top-left lands at (x0,y0)
        x = int(x0 - left)
        y = int(y0 - top)

        # Render: camera-like shadow or legacy stroke.
        if self.render_mode == "shadow":
            # Shadow first (slight offset), then main text.
            try:
                lo, hi = float(self.shadow_offset_frac_range[0]), float(self.shadow_offset_frac_range[1])
                lo, hi = (lo, hi) if hi >= lo else (hi, lo)
                off = max(1, int(round(font_size * random.uniform(max(0.0, lo), max(0.0, hi)))))
                shadow_alpha = self._sample_int_range(*self.shadow_alpha_range)
                shadow_fill = (self.shadow_color[0], self.shadow_color[1], self.shadow_color[2], self._clamp_int(shadow_alpha, 0, 255))
                draw.text((x + off, y + off), text, font=font, fill=shadow_fill)
            except Exception:
                pass
            draw.text((x, y), text, font=font, fill=fill)
        elif self.render_mode == "stroke":
            # Stroke to enhance visibility (legacy behavior).
            stroke_w = max(1, int(round(font_size * max(0.0, self.stroke_width_frac))))
            stroke_alpha = int(self._clamp_int(int(round(alpha * self.stroke_alpha_mult)), 0, 255))
            stroke_fill = (self.stroke_color[0], self.stroke_color[1], self.stroke_color[2], stroke_alpha)
            try:
                draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke_w, stroke_fill=stroke_fill)
            except Exception:
                draw.text((x, y), text, font=font, fill=fill)
        else:
            draw.text((x, y), text, font=font, fill=fill)

        out = Image.alpha_composite(rgba, overlay)
        return out.convert("RGB")


class RandomLightSpot:
    def __init__(
        self,
        p: float = 0.3,
        radius_frac_range: Tuple[float, float] = (0.06, 0.18),
        alpha_range: Tuple[float, float] = (0.2, 0.6),
        color: Tuple[int, int, int] = (255, 255, 220),
        blur_frac: float = 0.5,
        *,
        # New (optional): more realistic flare-style blending that preserves texture.
        # - composite: legacy behavior (blend toward a flat color)
        # - screen/additive: flare-style brighten/tint using ImageChops + mask
        blend_mode: str = "composite",
        # Optional center bias in normalized coordinates [0..1]
        center_x_frac_range: Tuple[float, float] = (0.0, 1.0),
        center_y_frac_range: Tuple[float, float] = (0.0, 1.0),
        # Optional: core+halo control (only used for non-composite blend modes)
        core_blur_frac: Optional[float] = None,
        halo_enabled: bool = True,
        halo_scale_range: Tuple[float, float] = (2.0, 3.5),
        halo_alpha_mult_range: Tuple[float, float] = (0.12, 0.35),
        halo_blur_frac: Optional[float] = None,
        # Grid (multi-spot) mode: create a rows x cols arrangement of spots.
        grid_enabled: bool = False,
        grid_rows_range: Tuple[int, int] = (1, 1),
        grid_cols_range: Tuple[int, int] = (1, 1),
        # Spacing as a multiple of the *core radius* (r). Example: 6.0 => centers ~6r apart.
        grid_spacing_mul_range: Tuple[float, float] = (5.0, 9.0),
        # Random jitter as a fraction of spacing (0..1).
        grid_jitter_frac: float = 0.12,
        # Per-spot dropout (skip a cell), useful for imperfect grids.
        grid_cell_dropout_prob: float = 0.0,
        # When true, all cells share the same sampled radius/alpha; when false, sample per cell.
        grid_share_params: bool = True,
    ) -> None:
        self.p = float(p)
        self.radius_frac_range = (float(radius_frac_range[0]), float(radius_frac_range[1]))
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.color = (int(color[0]), int(color[1]), int(color[2]))
        self.blur_frac = float(blur_frac)
        self.blend_mode = str(blend_mode or "composite").strip().lower()
        self.center_x_frac_range = (float(center_x_frac_range[0]), float(center_x_frac_range[1]))
        self.center_y_frac_range = (float(center_y_frac_range[0]), float(center_y_frac_range[1]))
        self.core_blur_frac = None if core_blur_frac is None else float(core_blur_frac)
        self.halo_enabled = bool(halo_enabled)
        self.halo_scale_range = (float(halo_scale_range[0]), float(halo_scale_range[1]))
        self.halo_alpha_mult_range = (float(halo_alpha_mult_range[0]), float(halo_alpha_mult_range[1]))
        self.halo_blur_frac = None if halo_blur_frac is None else float(halo_blur_frac)
        self.grid_enabled = bool(grid_enabled)
        self.grid_rows_range = (int(grid_rows_range[0]), int(grid_rows_range[1]))
        self.grid_cols_range = (int(grid_cols_range[0]), int(grid_cols_range[1]))
        self.grid_spacing_mul_range = (float(grid_spacing_mul_range[0]), float(grid_spacing_mul_range[1]))
        self.grid_jitter_frac = float(grid_jitter_frac)
        self.grid_cell_dropout_prob = float(grid_cell_dropout_prob)
        self.grid_share_params = bool(grid_share_params)

    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))

    @classmethod
    def _sample_center(cls, size: int, frac_range: Tuple[float, float]) -> int:
        if size <= 1:
            return 0
        lo_f, hi_f = float(frac_range[0]), float(frac_range[1])
        lo = int(round(cls._clamp01(min(lo_f, hi_f)) * (size - 1)))
        hi = int(round(cls._clamp01(max(lo_f, hi_f)) * (size - 1)))
        lo = max(0, min(size - 1, lo))
        hi = max(lo, min(size - 1, hi))
        return random.randint(lo, hi)

    @staticmethod
    def _blur_radius(r: int, frac: float) -> int:
        if not (r > 0):
            return 0
        if frac <= 0.0:
            return 0
        return max(1, int(r * frac))

    def _build_mask_legacy(self, *, w: int, h: int, cx: int, cy: int, r: int, alpha_val: int) -> Image.Image:
        # Legacy: solid disk then blur (single scale)
        mask = Image.new("L", (w, h), 0)
        draw_mask = ImageDraw.Draw(mask)
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw_mask.ellipse(bbox, fill=int(alpha_val))
        blur_radius = self._blur_radius(r, float(self.blur_frac))
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        return mask

    def _build_mask_core_halo(
        self,
        *,
        w: int,
        h: int,
        cx: int,
        cy: int,
        r_core: int,
        alpha_core: int,
    ) -> Image.Image:
        # Flare-style: strong core + softer halo, combined and clamped to [0,255].
        core_blur = float(self.core_blur_frac) if self.core_blur_frac is not None else float(self.blur_frac) * 0.25
        halo_blur = float(self.halo_blur_frac) if self.halo_blur_frac is not None else float(self.blur_frac)

        # Core
        m_core = Image.new("L", (w, h), 0)
        d_core = ImageDraw.Draw(m_core)
        d_core.ellipse([cx - r_core, cy - r_core, cx + r_core, cy + r_core], fill=int(alpha_core))
        br_core = self._blur_radius(r_core, core_blur)
        if br_core > 0:
            m_core = m_core.filter(ImageFilter.GaussianBlur(radius=br_core))

        if not self.halo_enabled:
            return m_core

        # Halo
        lo_s, hi_s = self.halo_scale_range
        scale = float(random.uniform(min(lo_s, hi_s), max(lo_s, hi_s)))
        r_halo = max(r_core + 1, int(r_core * max(1.1, scale)))
        lo_m, hi_m = self.halo_alpha_mult_range
        mult = float(random.uniform(min(lo_m, hi_m), max(lo_m, hi_m)))
        alpha_halo = int(max(0, min(255, int(alpha_core * mult))))
        m_halo = Image.new("L", (w, h), 0)
        d_halo = ImageDraw.Draw(m_halo)
        d_halo.ellipse([cx - r_halo, cy - r_halo, cx + r_halo, cy + r_halo], fill=int(alpha_halo))
        br_halo = self._blur_radius(r_halo, halo_blur)
        if br_halo > 0:
            m_halo = m_halo.filter(ImageFilter.GaussianBlur(radius=br_halo))

        # Combine (clamped)
        return ImageChops.add(m_halo, m_core)

    def _sample_grid_dims(self) -> Tuple[int, int]:
        r0, r1 = self.grid_rows_range
        c0, c1 = self.grid_cols_range
        r_lo, r_hi = (min(r0, r1), max(r0, r1))
        c_lo, c_hi = (min(c0, c1), max(c0, c1))
        rows = max(1, int(random.randint(max(1, r_lo), max(1, r_hi))))
        cols = max(1, int(random.randint(max(1, c_lo), max(1, c_hi))))
        return rows, cols

    def _sample_spacing_px(self, *, r_core: int) -> Tuple[float, float]:
        lo, hi = self.grid_spacing_mul_range
        lo, hi = (min(lo, hi), max(lo, hi))
        mul_x = float(random.uniform(lo, hi))
        mul_y = float(random.uniform(lo, hi))
        s = max(1.0, float(r_core) * max(1e-3, mul_x))
        t = max(1.0, float(r_core) * max(1e-3, mul_y))
        return s, t

    def _grid_origin_and_spacing(
        self,
        *,
        w: int,
        h: int,
        rows: int,
        cols: int,
        r_core: int,
        spacing_x: float,
        spacing_y: float,
    ) -> Tuple[float, float, float, float]:
        """
        Choose a top-left center (x0,y0) for the grid and adjust spacing to fit in-bounds.
        Centers are constrained to [r, w-1-r] and [r, h-1-r].
        """
        # Clamp spacing to fit
        if cols > 1:
            max_sx = float(max(1.0, (w - 2 * r_core) / float(cols - 1)))
            spacing_x = min(spacing_x, max_sx)
        if rows > 1:
            max_sy = float(max(1.0, (h - 2 * r_core) / float(rows - 1)))
            spacing_y = min(spacing_y, max_sy)

        grid_w = float((cols - 1) * spacing_x) if cols > 1 else 0.0
        grid_h = float((rows - 1) * spacing_y) if rows > 1 else 0.0

        x_min = float(r_core)
        y_min = float(r_core)
        x_max = float(max(r_core, (w - 1 - r_core))) - grid_w
        y_max = float(max(r_core, (h - 1 - r_core))) - grid_h
        if x_max < x_min:
            x_max = x_min
        if y_max < y_min:
            y_max = y_min
        x0 = float(random.uniform(x_min, x_max))
        y0 = float(random.uniform(y_min, y_max))
        return x0, y0, spacing_x, spacing_y

    def _build_multi_spot_mask(
        self,
        *,
        w: int,
        h: int,
        rows: int,
        cols: int,
        r_core_base: int,
        alpha_base: int,
        mode: str,
    ) -> Image.Image:
        combined = Image.new("L", (w, h), 0)

        # Determine spacing/origin for the grid
        if rows > 1 or cols > 1:
            sx, sy = self._sample_spacing_px(r_core=r_core_base)
        else:
            sx, sy = 0.0, 0.0
        x0, y0, sx, sy = self._grid_origin_and_spacing(
            w=w, h=h, rows=rows, cols=cols, r_core=r_core_base, spacing_x=sx, spacing_y=sy
        )
        jitter = max(0.0, min(1.0, float(self.grid_jitter_frac)))
        jitter_px_x = abs(float(sx)) * jitter if cols > 1 else float(r_core_base) * jitter
        jitter_px_y = abs(float(sy)) * jitter if rows > 1 else float(r_core_base) * jitter

        # At least one spot should survive dropout.
        kept_any = False

        for rr in range(rows):
            for cc in range(cols):
                if self.grid_cell_dropout_prob > 0.0 and random.random() < self.grid_cell_dropout_prob:
                    continue

                if self.grid_share_params:
                    r_core = int(r_core_base)
                    alpha_val = int(alpha_base)
                else:
                    r_core = int(min(w, h) * random.uniform(*self.radius_frac_range))
                    alpha_val = int(255 * random.uniform(*self.alpha_range))

                cx = x0 + float(cc) * float(sx)
                cy = y0 + float(rr) * float(sy)
                if jitter_px_x > 0:
                    cx += random.uniform(-jitter_px_x, jitter_px_x)
                if jitter_px_y > 0:
                    cy += random.uniform(-jitter_px_y, jitter_px_y)
                cxi = int(round(cx))
                cyi = int(round(cy))

                # Ensure within bounds
                cxi = max(r_core, min(w - 1 - r_core, cxi))
                cyi = max(r_core, min(h - 1 - r_core, cyi))

                if mode in ("composite", "legacy", "replace"):
                    spot_mask = self._build_mask_legacy(
                        w=w, h=h, cx=cxi, cy=cyi, r=r_core, alpha_val=alpha_val
                    )
                else:
                    spot_mask = self._build_mask_core_halo(
                        w=w,
                        h=h,
                        cx=cxi,
                        cy=cyi,
                        r_core=r_core,
                        alpha_core=alpha_val,
                    )
                combined = ImageChops.add(combined, spot_mask)
                kept_any = True

        if (not kept_any) and (rows * cols > 0):
            # Force one spot in the center of the grid
            cx = int(round(x0 + float(max(0, cols - 1)) * float(sx) * 0.5))
            cy = int(round(y0 + float(max(0, rows - 1)) * float(sy) * 0.5))
            cx = max(r_core_base, min(w - 1 - r_core_base, cx))
            cy = max(r_core_base, min(h - 1 - r_core_base, cy))
            if mode in ("composite", "legacy", "replace"):
                combined = self._build_mask_legacy(
                    w=w, h=h, cx=cx, cy=cy, r=r_core_base, alpha_val=alpha_base
                )
            else:
                combined = self._build_mask_core_halo(
                    w=w, h=h, cx=cx, cy=cy, r_core=r_core_base, alpha_core=alpha_base
                )

        return combined

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0:
            return img
        if torch.rand(()) > self.p:
            return img
        if not isinstance(img, Image.Image):
            return img
        w, h = img.size
        r = int(min(w, h) * random.uniform(*self.radius_frac_range))
        cx = self._sample_center(w, self.center_x_frac_range)
        cy = self._sample_center(h, self.center_y_frac_range)
        alpha_val = int(255 * random.uniform(*self.alpha_range))

        mode = str(self.blend_mode or "composite").strip().lower()
        # If grid mode is enabled, ignore single (cx,cy) and build a multi-spot mask.
        if self.grid_enabled:
            rows, cols = self._sample_grid_dims()
            mask = self._build_multi_spot_mask(
                w=w,
                h=h,
                rows=rows,
                cols=cols,
                r_core_base=r,
                alpha_base=alpha_val,
                mode=mode,
            )
        else:
            if mode in ("composite", "legacy", "replace"):
                mask = self._build_mask_legacy(w=w, h=h, cx=cx, cy=cy, r=r, alpha_val=alpha_val)
            else:
                mask = self._build_mask_core_halo(w=w, h=h, cx=cx, cy=cy, r_core=r, alpha_core=alpha_val)

        if mode in ("composite", "legacy", "replace"):
            rgba = img.convert("RGBA")
            spot_layer = Image.new("RGBA", (w, h), self.color + (0,))
            composite = Image.composite(spot_layer, rgba, mask)
            return composite.convert("RGB")

        # Flare-style blend: preserve underlying texture, brighten/tint via screen/add.
        base = img.convert("RGB")
        overlay = Image.new("RGB", (w, h), self.color)
        if mode in ("add", "additive", "plus"):
            effected = ImageChops.add(base, overlay, scale=1.0, offset=0)
        else:
            # default: screen (more lens-flare-like than hard replace)
            effected = ImageChops.screen(base, overlay)
        out = Image.composite(effected, base, mask)
        return out.convert("RGB")


def _maybe_add_color_jitter(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    """
    Optionally add ColorJitter based on config.

    Expected config structure under `data.augment`:

    color_jitter:
      enabled: true
      prob: 0.8
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.02
    """
    cj_cfg: Dict[str, Any] = dict(cfg.get("color_jitter", {}))
    if not bool(cj_cfg.get("enabled", False)):
        return

    p = float(cj_cfg.get("prob", 0.8))
    brightness = cj_cfg.get("brightness", 0.0)
    contrast = cj_cfg.get("contrast", 0.0)
    saturation = cj_cfg.get("saturation", 0.0)
    hue = cj_cfg.get("hue", 0.0)

    # If all parameters are zero/None, skip to avoid a no-op transform.
    if not any(x for x in [brightness, contrast, saturation, hue]):
        return

    jitter = T.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )
    # Apply with probability p on PIL images before ToTensor.
    transforms_list.append(T.RandomApply([jitter], p=p))


def _maybe_add_affine(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    affine_cfg: Dict[str, Any] = dict(cfg.get("random_affine", {}))
    if not bool(affine_cfg.get("enabled", False)):
        return
    degrees = affine_cfg.get("degrees", 0.0)
    translate = affine_cfg.get("translate", None)
    scale = affine_cfg.get("scale", None)
    shear = affine_cfg.get("shear", None)
    interpolation = getattr(T.InterpolationMode, str(affine_cfg.get("interpolation", "bilinear")).upper(), T.InterpolationMode.BILINEAR)
    fill = affine_cfg.get("fill", 0)
    transforms_list.append(
        T.RandomAffine(
            degrees=degrees,
            translate=tuple(translate) if translate is not None else None,
            scale=tuple(scale) if scale is not None else None,
            shear=tuple(shear) if shear is not None else None,
            interpolation=interpolation,
            fill=fill,
        )
    )


def _maybe_add_vertical_flip(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    vflip_cfg: Dict[str, Any] = dict(cfg.get("vertical_flip", {}))
    if not bool(vflip_cfg.get("enabled", False)):
        return
    p = float(vflip_cfg.get("prob", 0.0))
    if p > 0.0:
        transforms_list.append(T.RandomVerticalFlip(p=p))


def _maybe_add_blur_and_noise(cfg: Dict[str, Any], before_normalize: List[Any], after_tensor: List[Any]) -> None:
    blur_cfg: Dict[str, Any] = dict(cfg.get("gaussian_blur", {}))
    if bool(blur_cfg.get("enabled", False)):
        p = float(blur_cfg.get("prob", 0.5))
        k_range = blur_cfg.get("kernel_size_range", blur_cfg.get("kernel_size", [3, 5]))
        sigma = blur_cfg.get("sigma", [0.1, 1.0])
        # Apply after ToTensor for consistent tensor-domain operation
        after_tensor.append(
            ProbGaussianBlur(
                p=p,
                kernel_size_range=(int(k_range[0]), int(k_range[1])) if isinstance(k_range, (list, tuple)) else (3, int(k_range)),
                sigma=tuple(sigma) if isinstance(sigma, (list, tuple)) else float(sigma),
            )
        )

    noise_cfg: Dict[str, Any] = dict(cfg.get("gaussian_noise", {}))
    if bool(noise_cfg.get("enabled", False)):
        mean = float(noise_cfg.get("mean", 0.0))
        std = float(noise_cfg.get("std", 0.01))
        p = float(noise_cfg.get("prob", 0.5))
        after_tensor.append(AddGaussianNoise(mean=mean, std=std, p=p))


def _maybe_add_random_erasing(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    er_cfg: Dict[str, Any] = dict(cfg.get("random_erasing", {}))
    if not bool(er_cfg.get("enabled", False)):
        return
    p = float(er_cfg.get("p", 0.25))
    scale = tuple(er_cfg.get("scale", [0.02, 0.1]))
    ratio = tuple(er_cfg.get("ratio", [0.3, 3.3]))
    value = er_cfg.get("value", "random")
    transforms_list.append(T.RandomErasing(p=p, scale=scale, ratio=ratio, value=value))


def _maybe_add_visual_overlays(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    wm_cfg: Dict[str, Any] = dict(cfg.get("watermark", {}))
    if bool(wm_cfg.get("enabled", False)):
        # Backward compatible: support both "style" and legacy "mode" key naming.
        style = wm_cfg.get("style", wm_cfg.get("mode", "random"))
        pos_choices = list(wm_cfg.get("position_choices", wm_cfg.get("anchor_choices", [])) or [])
        pos_probs = list(wm_cfg.get("position_probs", wm_cfg.get("anchor_probs", [])) or [])
        transforms_list.append(
            RandomWatermark(
                p=float(wm_cfg.get("prob", 0.3)),
                texts=list(wm_cfg.get("texts", [])) or None,
                timestamp_prob=float(wm_cfg.get("timestamp_prob", 0.5)),
                timestamp_format=str(wm_cfg.get("timestamp_format", wm_cfg.get("timestamp_fmt", "%Y-%m-%d %H:%M:%S"))),
                font_path=wm_cfg.get("font_path", None),
                font_size_frac_range=tuple(wm_cfg.get("font_size_frac_range", [0.04, 0.12])),
                alpha_range=tuple(wm_cfg.get("alpha_range", [128, 200])),
                color_choices=list(wm_cfg.get("color_choices", [])) or None,
                color_jitter=int(wm_cfg.get("color_jitter", 0)),
                use_random_text=bool(wm_cfg.get("use_random_text", False)),
                random_text_length_range=tuple(wm_cfg.get("random_text_length_range", [6, 12])),
                charset=str(wm_cfg.get("charset", "")) if wm_cfg.get("charset", None) is not None else None,
                style=str(style),
                position=str(wm_cfg.get("position", wm_cfg.get("anchor", "random"))),
                position_choices=pos_choices or None,
                position_probs=pos_probs or None,
                margin_frac_range=tuple(wm_cfg.get("margin_frac_range", [0.01, 0.04])),
                render_mode=wm_cfg.get("render_mode", None),
                shadow_color=tuple(wm_cfg.get("shadow_color", [0, 0, 0])),
                shadow_alpha_range=tuple(wm_cfg.get("shadow_alpha_range", [140, 255])),
                shadow_offset_frac_range=tuple(wm_cfg.get("shadow_offset_frac_range", [0.03, 0.07])),
                stroke_color=tuple(wm_cfg.get("stroke_color", [0, 0, 0])),
                stroke_alpha_mult=float(wm_cfg.get("stroke_alpha_mult", 1.0)),
                stroke_width_frac=float(wm_cfg.get("stroke_width_frac", 1.0 / 15.0)),
            )
        )

    ls_cfg: Dict[str, Any] = dict(cfg.get("light_spot", {}))
    if bool(ls_cfg.get("enabled", False)):
        # Support both legacy params and newer flare-style options.
        # New keys are optional and default to safe values in RandomLightSpot.
        blend_mode = str(ls_cfg.get("blend_mode", ls_cfg.get("mode", "composite")) or "composite")
        cx_rng = ls_cfg.get("center_x_frac_range", [0.0, 1.0])
        cy_rng = ls_cfg.get("center_y_frac_range", [0.0, 1.0])
        core_blur_frac = ls_cfg.get("core_blur_frac", None)
        halo_enabled = ls_cfg.get("halo_enabled", True)
        halo_scale_range = ls_cfg.get("halo_scale_range", [2.0, 3.5])
        halo_alpha_mult_range = ls_cfg.get("halo_alpha_mult_range", [0.12, 0.35])
        halo_blur_frac = ls_cfg.get("halo_blur_frac", None)
        grid_enabled = bool(ls_cfg.get("grid_enabled", False))
        grid_rows_range = ls_cfg.get("grid_rows_range", [1, 1])
        grid_cols_range = ls_cfg.get("grid_cols_range", [1, 1])
        grid_spacing_mul_range = ls_cfg.get("grid_spacing_mul_range", [5.0, 9.0])
        grid_jitter_frac = float(ls_cfg.get("grid_jitter_frac", 0.12))
        grid_cell_dropout_prob = float(ls_cfg.get("grid_cell_dropout_prob", 0.0))
        grid_share_params = bool(ls_cfg.get("grid_share_params", True))
        transforms_list.append(
            RandomLightSpot(
                p=float(ls_cfg.get("prob", 0.3)),
                radius_frac_range=tuple(ls_cfg.get("radius_frac_range", [0.06, 0.18])),
                alpha_range=tuple(ls_cfg.get("alpha_range", [0.2, 0.6])),
                color=tuple(ls_cfg.get("color", [255, 255, 220])),
                blur_frac=float(ls_cfg.get("blur_frac", 0.5)),
                blend_mode=blend_mode,
                center_x_frac_range=tuple(cx_rng),
                center_y_frac_range=tuple(cy_rng),
                core_blur_frac=float(core_blur_frac) if core_blur_frac is not None else None,
                halo_enabled=bool(halo_enabled),
                halo_scale_range=tuple(halo_scale_range),
                halo_alpha_mult_range=tuple(halo_alpha_mult_range),
                halo_blur_frac=float(halo_blur_frac) if halo_blur_frac is not None else None,
                grid_enabled=grid_enabled,
                grid_rows_range=tuple(grid_rows_range),
                grid_cols_range=tuple(grid_cols_range),
                grid_spacing_mul_range=tuple(grid_spacing_mul_range),
                grid_jitter_frac=grid_jitter_frac,
                grid_cell_dropout_prob=grid_cell_dropout_prob,
                grid_share_params=grid_share_params,
            )
        )


def build_train_transform(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
    augment_cfg: Optional[Dict[str, Any]] = None,
) -> T.Compose:
    cfg = dict(augment_cfg or {})

    # Optional augmentation policy switch.
    #
    # - legacy (default): current handcrafted pipeline (flip/affine/jitter/blur/noise/erasing + overlays)
    # - augmix          : AugMix (single-output tensor, no JSD tuple) + optional watermark/light_spot
    policy = str(cfg.get("policy", cfg.get("augment_policy", "legacy")) or "legacy").strip().lower()
    if policy in {"augmix", "aug_mix", "aug-mix"}:
        return build_train_transform_augmix(
            image_size=image_size,
            mean=mean,
            std=std,
            augment_cfg=cfg,
        )

    # 1) Geometric and color transforms on PIL images
    pre_tensor: List[Any] = []
    # RandomResizedCrop (optional; disabled by default).
    #
    # Why disabled by default:
    #   - For highly rectangular images (e.g. 2:1 CSIRO), torchvision's default ratio range
    #     (3/4..4/3) + a high scale (e.g. >=0.8) can make it *impossible* to sample a crop.
    #     In that case, RandomResizedCrop silently falls back to a deterministic center-crop.
    #
    rrc_enabled = bool(
        cfg.get(
            "random_resized_crop_enabled",
            bool(dict(cfg.get("random_resized_crop", {})).get("enabled", False)),
        )
    )
    if rrc_enabled:
        rrc_scale = cfg.get(
            "random_resized_crop_scale",
            dict(cfg.get("random_resized_crop", {})).get("scale", [0.8, 1.0]),
        )
        # Optional override: allow specifying ratio range in YAML (tuple/list of two floats).
        rrc_ratio = cfg.get(
            "random_resized_crop_ratio",
            dict(cfg.get("random_resized_crop", {})).get("ratio", None),
        )
        if rrc_ratio is None:
            pre_tensor.append(T.RandomResizedCrop(image_size, scale=tuple(rrc_scale)))
        else:
            pre_tensor.append(
                T.RandomResizedCrop(
                    image_size, scale=tuple(rrc_scale), ratio=tuple(rrc_ratio)
                )
            )
    else:
        # Safe default: match eval pipeline sizing without introducing random crops.
        pre_tensor.append(
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC)
        )

    hflip_prob = float(cfg.get("horizontal_flip_prob", cfg.get("horizontal_flip", {}).get("prob", 0.5)))
    if hflip_prob > 0.0:
        pre_tensor.append(T.RandomHorizontalFlip(p=hflip_prob))

    _maybe_add_vertical_flip(cfg, pre_tensor)
    _maybe_add_affine(cfg, pre_tensor)
    _maybe_add_color_jitter(cfg, pre_tensor)
    _maybe_add_visual_overlays(cfg, pre_tensor)

    # 2) Convert to tensor for tensor-domain ops
    to_tensor_and_pre_norm: List[Any] = [T.ToTensor()]
    _maybe_add_blur_and_noise(cfg, before_normalize=to_tensor_and_pre_norm, after_tensor=to_tensor_and_pre_norm)

    # 3) Normalize
    normalize_and_erasing: List[Any] = [T.Normalize(mean=mean, std=std)]
    _maybe_add_random_erasing(cfg, normalize_and_erasing)

    aug_tf = T.Compose(pre_tensor + to_tensor_and_pre_norm + normalize_and_erasing)

    # Optional: per-sample probability to bypass augmentation and use clean eval pipeline
    no_aug_prob = float(cfg.get("no_augment_prob", cfg.get("clean_sample_prob", 0.0)))
    if no_aug_prob <= 0.0:
        return aug_tf

    clean_tf = build_eval_transform(image_size=image_size, mean=mean, std=std)

    class ChoiceByProb:
        def __init__(self, p_clean: float, clean: Any, aug: Any) -> None:
            self.p_clean = float(p_clean)
            self.clean = clean
            self.aug = aug

        def __call__(self, x: Any) -> Any:
            if torch.rand(()) < self.p_clean:
                return self.clean(x)
            return self.aug(x)

    return ChoiceByProb(no_aug_prob, clean_tf, aug_tf)


def build_train_transform_augmix(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
    augment_cfg: Optional[Dict[str, Any]] = None,
) -> T.Compose:
    """
    AugMix-based training transform (single output tensor).

    Design goals:
      - Keep dataloader / training step unchanged: dataset returns a single `Tensor[C,H,W]`.
      - Avoid stacking legacy "handcrafted" transforms on top of AugMix by default.
      - Preserve optional visual overlays (watermark / light_spot) from config.
    """
    cfg = dict(augment_cfg or {})
    augmix_cfg: Dict[str, Any] = dict(cfg.get("augmix", {}) or {})

    # If explicitly disabled, fall back to the clean eval pipeline.
    if not bool(augmix_cfg.get("enabled", True)):
        return build_eval_transform(image_size=image_size, mean=mean, std=std)

    # 1) Base PIL transforms (size + optional hflip) + optional overlays.
    pre_pil: List[Any] = []

    rrc_enabled = bool(
        cfg.get(
            "random_resized_crop_enabled",
            bool(dict(cfg.get("random_resized_crop", {})).get("enabled", False)),
        )
    )
    if rrc_enabled:
        rrc_scale = cfg.get(
            "random_resized_crop_scale",
            dict(cfg.get("random_resized_crop", {})).get("scale", [0.8, 1.0]),
        )
        rrc_ratio = cfg.get(
            "random_resized_crop_ratio",
            dict(cfg.get("random_resized_crop", {})).get("ratio", None),
        )
        if rrc_ratio is None:
            pre_pil.append(T.RandomResizedCrop(image_size, scale=tuple(rrc_scale)))
        else:
            pre_pil.append(
                T.RandomResizedCrop(image_size, scale=tuple(rrc_scale), ratio=tuple(rrc_ratio))
            )
    else:
        pre_pil.append(T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC))

    # Keep horizontal flip as a simple, widely-used geometric augmentation (tunable via YAML).
    hflip_prob = float(cfg.get("horizontal_flip_prob", cfg.get("horizontal_flip", {}).get("prob", 0.5)))
    if hflip_prob > 0.0:
        pre_pil.append(T.RandomHorizontalFlip(p=hflip_prob))

    # Preserve these two augmentations (explicitly favored in the discussion).
    _maybe_add_visual_overlays(cfg, pre_pil)

    # 2) AugMix -> returns a normalized tensor.
    from .augmix import AugMix, AugMixConfig

    preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    am_cfg = AugMixConfig(
        severity=int(augmix_cfg.get("severity", augmix_cfg.get("aug_severity", 3))),
        width=int(augmix_cfg.get("width", augmix_cfg.get("mixture_width", 3))),
        depth=int(augmix_cfg.get("depth", augmix_cfg.get("mixture_depth", -1))),
        alpha=float(augmix_cfg.get("alpha", augmix_cfg.get("aug_prob_coeff", 1.0))),
        all_ops=bool(augmix_cfg.get("all_ops", augmix_cfg.get("all", False))),
    )
    augmix_tf = AugMix(preprocess, cfg=am_cfg)
    aug_tf = T.Compose([*pre_pil, augmix_tf])

    # Optional: per-sample probability to bypass augmentation and use clean eval pipeline
    no_aug_prob = float(cfg.get("no_augment_prob", cfg.get("clean_sample_prob", 0.0)))
    if no_aug_prob <= 0.0:
        return aug_tf

    clean_tf = build_eval_transform(image_size=image_size, mean=mean, std=std)

    class ChoiceByProb:
        def __init__(self, p_clean: float, clean: Any, aug: Any) -> None:
            self.p_clean = float(p_clean)
            self.clean = clean
            self.aug = aug

        def __call__(self, x: Any) -> Any:
            if torch.rand(()) < self.p_clean:
                return self.clean(x)
            return self.aug(x)

    return ChoiceByProb(no_aug_prob, clean_tf, aug_tf)


def build_eval_transform(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
) -> T.Compose:
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def build_transforms(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
    augment_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[T.Compose, T.Compose]:
    train_tf = build_train_transform(image_size=image_size, mean=mean, std=std, augment_cfg=augment_cfg)
    val_tf = build_eval_transform(image_size=image_size, mean=mean, std=std)
    return train_tf, val_tf


