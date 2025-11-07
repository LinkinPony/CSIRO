import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchvision.transforms as T


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
        k = blur_cfg.get("kernel_size", 3)
        sigma = blur_cfg.get("sigma", [0.1, 1.0])
        before_normalize.append(T.GaussianBlur(kernel_size=int(k), sigma=tuple(sigma) if isinstance(sigma, (list, tuple)) else sigma))

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

    # 2) Convert to tensor for tensor-domain ops
    to_tensor_and_pre_norm: List[Any] = [T.ToTensor()]
    _maybe_add_blur_and_noise(cfg, before_normalize=to_tensor_and_pre_norm, after_tensor=to_tensor_and_pre_norm)

    # 3) Normalize
    normalize_and_erasing: List[Any] = [T.Normalize(mean=mean, std=std)]
    _maybe_add_random_erasing(cfg, normalize_and_erasing)

    return T.Compose(pre_tensor + to_tensor_and_pre_norm + normalize_and_erasing)


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


