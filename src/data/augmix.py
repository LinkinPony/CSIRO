from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
from PIL import Image, ImageEnhance, ImageOps


def _int_parameter(level: float, maxval: float) -> int:
    """Scale `level` in [0, 10] to an int in [0, maxval]."""
    return int(level * float(maxval) / 10.0)


def _float_parameter(level: float, maxval: float) -> float:
    """Scale `level` in [0, 10] to a float in [0, maxval]."""
    return float(level) * float(maxval) / 10.0


def _sample_level(n: int) -> float:
    # Match the reference: uniform in [0.1, n].
    n_i = max(1, int(n))
    return random.uniform(0.1, float(n_i))


def _resample_bilinear():
    # Pillow>=9 uses Image.Resampling; older versions use Image.BILINEAR.
    try:
        return Image.Resampling.BILINEAR  # type: ignore[attr-defined]
    except Exception:
        return Image.BILINEAR


def autocontrast(pil_img: Image.Image, _severity: int) -> Image.Image:
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img: Image.Image, _severity: int) -> Image.Image:
    return ImageOps.equalize(pil_img)


def posterize(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _int_parameter(_sample_level(severity), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img: Image.Image, severity: int) -> Image.Image:
    degrees = _int_parameter(_sample_level(severity), 30)
    if random.random() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=_resample_bilinear())


def solarize(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _int_parameter(_sample_level(severity), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 0.3)
    if random.random() > 0.5:
        level = -level
    w, h = pil_img.size
    return pil_img.transform(
        (w, h),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=_resample_bilinear(),
    )


def shear_y(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 0.3)
    if random.random() > 0.5:
        level = -level
    w, h = pil_img.size
    return pil_img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=_resample_bilinear(),
    )


def translate_x(pil_img: Image.Image, severity: int) -> Image.Image:
    w, h = pil_img.size
    level = _int_parameter(_sample_level(severity), w / 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=_resample_bilinear(),
    )


def translate_y(pil_img: Image.Image, severity: int) -> Image.Image:
    w, h = pil_img.size
    level = _int_parameter(_sample_level(severity), h / 3)
    if random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=_resample_bilinear(),
    )


def color(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


def contrast(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


def brightness(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


def sharpness(pil_img: Image.Image, severity: int) -> Image.Image:
    level = _float_parameter(_sample_level(severity), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


AUGMIX_OPS_BASE = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]


AUGMIX_OPS_ALL = [
    *AUGMIX_OPS_BASE,
    color,
    contrast,
    brightness,
    sharpness,
]


@dataclass(frozen=True)
class AugMixConfig:
    severity: int = 3
    width: int = 3
    depth: int = -1  # -1 => random depth in [1, 3]
    alpha: float = 1.0
    all_ops: bool = False


class AugMix:
    """
    Minimal AugMix implementation (single-output tensor).

    This matches the reference algorithm, but supports arbitrary image sizes
    (no hard-coded IMAGE_SIZE) and outputs a single tensor (no JSD tuple).
    """

    def __init__(
        self,
        preprocess: Callable[[Image.Image], torch.Tensor],
        *,
        cfg: AugMixConfig,
    ) -> None:
        self.preprocess = preprocess
        self.cfg = cfg

    def _sample_ws(self, width: int, alpha: float) -> torch.Tensor:
        # Dirichlet([alpha] * width)
        a = float(alpha)
        a = 1e-6 if not (a > 0.0) else a
        conc = torch.full((int(width),), a, dtype=torch.float32)
        return torch.distributions.Dirichlet(conc).sample()

    def _sample_m(self, alpha: float) -> torch.Tensor:
        a = float(alpha)
        a = 1e-6 if not (a > 0.0) else a
        return torch.distributions.Beta(a, a).sample()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            # Keep behavior consistent with other transforms: pass-through.
            return image  # type: ignore[return-value]

        severity = int(self.cfg.severity)
        severity = 1 if severity < 1 else (10 if severity > 10 else severity)

        width = int(self.cfg.width)
        width = 1 if width < 1 else width

        depth = int(self.cfg.depth)
        alpha = float(self.cfg.alpha)
        ops: Sequence[Callable[[Image.Image, int], Image.Image]] = (
            AUGMIX_OPS_ALL if bool(self.cfg.all_ops) else AUGMIX_OPS_BASE
        )

        ws = self._sample_ws(width=width, alpha=alpha)
        m = self._sample_m(alpha=alpha).to(dtype=torch.float32)

        clean = self.preprocess(image)
        mix = torch.zeros_like(clean)
        for i in range(width):
            image_aug = image.copy()
            d = depth if depth > 0 else random.randint(1, 3)
            for _ in range(int(d)):
                op = random.choice(list(ops))
                image_aug = op(image_aug, severity)
            mix = mix + ws[i].to(dtype=mix.dtype) * self.preprocess(image_aug)

        mixed = (1.0 - m) * clean + m * mix
        return mixed


