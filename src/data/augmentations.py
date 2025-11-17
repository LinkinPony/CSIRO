import math
import random
import string
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter


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
        font_path: Optional[str] = None,
        font_size_frac_range: Tuple[float, float] = (0.04, 0.12),
        alpha_range: Tuple[int, int] = (128, 200),
        color_choices: Optional[List[Tuple[int, int, int]]] = None,
        use_random_text: bool = False,
        random_text_length_range: Tuple[int, int] = (6, 12),
        charset: Optional[str] = None,
    ) -> None:
        self.p = float(p)
        self.texts = list(texts or ["CSIRO", "Biomass", "Train", "Sample"])
        self.timestamp_prob = float(timestamp_prob)
        self.font_path = font_path
        self.font_size_frac_range = (float(font_size_frac_range[0]), float(font_size_frac_range[1]))
        self.alpha_range = (int(alpha_range[0]), int(alpha_range[1]))
        self.color_choices = list(color_choices or [(255, 255, 255), (255, 230, 0), (0, 255, 255), (255, 128, 0)])
        self.use_random_text = bool(use_random_text)
        self.random_text_length_range = (int(random_text_length_range[0]), int(random_text_length_range[1]))
        default_charset = string.ascii_uppercase + string.digits
        self.charset = str(charset) if (charset and isinstance(charset, str) and len(charset) > 0) else default_charset

    def _choose_font(self, size: int) -> ImageFont.ImageFont:
        try:
            if self.font_path:
                return ImageFont.truetype(self.font_path, size=size)
        except Exception:
            pass
        try:
            return ImageFont.load_default()
        except Exception:
            return ImageFont.load_default()

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

        use_ts = random.random() < self.timestamp_prob
        if use_ts:
            text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
        alpha = int(random.randint(*self.alpha_range))
        fill = (int(color[0]), int(color[1]), int(color[2]), alpha)

        # Random position ensuring the text fits
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = font_size * len(text), int(font_size * 1.2)
        x = random.randint(0, max(0, w - tw)) if w > tw else 0
        y = random.randint(0, max(0, h - th)) if h > th else 0

        # Stroke to enhance visibility
        try:
            draw.text((x, y), text, font=font, fill=fill, stroke_width=max(1, font_size // 15), stroke_fill=(0, 0, 0, alpha))
        except Exception:
            draw.text((x, y), text, font=font, fill=fill)

        out = Image.alpha_composite(rgba, overlay)
        return out.convert("RGB")


class RandomLightSpot:
    def __init__(
        self,
        p: float = 0.3,
        radius_frac_range: Tuple[float, float] = (0.06, 0.18),
        alpha_range: Tuple[float, float] = (0.2, 0.6),
        color: Optional[Tuple[int, int, int]] = (255, 255, 220),
        blur_frac: float = 0.5,
    ) -> None:
        self.p = float(p)
        self.radius_frac_range = (float(radius_frac_range[0]), float(radius_frac_range[1]))
        self.alpha_range = (float(alpha_range[0]), float(alpha_range[1]))
        self.color = (
            (int(color[0]), int(color[1]), int(color[2]))
            if color is not None
            else None
        )
        self.blur_frac = float(blur_frac)
        self.use_random_color = False

    def enable_random_color(self, flag: bool = True) -> None:
        self.use_random_color = bool(flag)

    def _sample_color(self) -> Tuple[int, int, int]:
        # Default to a bright, slightly warm random color when randomization is enabled
        if self.use_random_color or self.color is None:
            r = random.randint(220, 255)
            g = random.randint(220, 255)
            b = random.randint(200, 255)
            return (r, g, b)
        return self.color

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.p <= 0.0:
            return img
        if torch.rand(()) > self.p:
            return img
        if not isinstance(img, Image.Image):
            return img
        w, h = img.size
        rgba = img.convert("RGBA")
        r_px = int(min(w, h) * random.uniform(*self.radius_frac_range))
        if r_px <= 0:
            return img

        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        alpha_max = int(255 * random.uniform(*self.alpha_range))
        if alpha_max <= 0:
            return img

        blur_radius = max(1, int(r_px * self.blur_frac))

        x0 = max(0, cx - r_px)
        x1 = min(w - 1, cx + r_px)
        y0 = max(0, cy - r_px)
        y1 = min(h - 1, cy + r_px)

        spot_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        pixels = spot_layer.load()
        color = self._sample_color()

        for y in range(y0, y1 + 1):
            dy = y - cy
            for x in range(x0, x1 + 1):
                dx = x - cx
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > r_px:
                    continue
                t = dist / float(r_px)
                # Smooth radial falloff; brighter in the center, fading towards the edges
                intensity = max(0.0, 1.0 - t * t)
                alpha = int(alpha_max * intensity)
                if alpha <= 0:
                    continue
                pixels[x, y] = (color[0], color[1], color[2], alpha)

        if blur_radius > 0:
            spot_layer = spot_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        composite = Image.alpha_composite(rgba, spot_layer)
        return composite.convert("RGB")


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
        transforms_list.append(
            RandomWatermark(
                p=float(wm_cfg.get("prob", 0.3)),
                texts=list(wm_cfg.get("texts", [])) or None,
                timestamp_prob=float(wm_cfg.get("timestamp_prob", 0.5)),
                font_path=wm_cfg.get("font_path", None),
                font_size_frac_range=tuple(wm_cfg.get("font_size_frac_range", [0.04, 0.12])),
                alpha_range=tuple(wm_cfg.get("alpha_range", [128, 200])),
                color_choices=list(wm_cfg.get("color_choices", [])) or None,
                use_random_text=bool(wm_cfg.get("use_random_text", False)),
                random_text_length_range=tuple(wm_cfg.get("random_text_length_range", [6, 12])),
                charset=str(wm_cfg.get("charset", "")) if wm_cfg.get("charset", None) is not None else None,
            )
        )

    ls_cfg: Dict[str, Any] = dict(cfg.get("light_spot", {}))
    if bool(ls_cfg.get("enabled", False)):
        transforms_list.append(
            RandomLightSpot(
                p=float(ls_cfg.get("prob", 0.3)),
                radius_frac_range=tuple(ls_cfg.get("radius_frac_range", [0.06, 0.18])),
                alpha_range=tuple(ls_cfg.get("alpha_range", [0.2, 0.6])),
                color=tuple(ls_cfg.get("color", [255, 255, 220])) if "color" in ls_cfg else None,
                blur_frac=float(ls_cfg.get("blur_frac", 0.5)),
            )
        )


def _maybe_add_color_jitter(cfg: Dict[str, Any], transforms_list: List[Any]) -> None:
    cj_cfg: Dict[str, Any] = dict(cfg.get("color_jitter", {}))
    if not bool(cj_cfg.get("enabled", False)):
        return
    brightness = cj_cfg.get("brightness", 0.0)
    contrast = cj_cfg.get("contrast", 0.0)
    saturation = cj_cfg.get("saturation", 0.0)
    hue = cj_cfg.get("hue", 0.0)
    transforms_list.append(
        T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )
    )


def build_train_transform(
    image_size: Union[int, Tuple[int, int]],
    mean: Sequence[float],
    std: Sequence[float],
    augment_cfg: Optional[Dict[str, Any]] = None,
) -> T.Compose:
    cfg = dict(augment_cfg or {})

    # 1) Geometric and color transforms on PIL images
    pre_tensor: List[Any] = []
    # Keep existing random resized crop + horizontal flip (backward-compat)
    rrc_scale = cfg.get("random_resized_crop_scale", cfg.get("random_resized_crop", {}).get("scale", [0.8, 1.0]))
    pre_tensor.append(T.RandomResizedCrop(image_size, scale=tuple(rrc_scale)))

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


