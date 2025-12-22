from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple


class RatioHeadMode(str, Enum):
    """
    How the biomass ratio task head is coupled to the main reg3 head.
    """

    SHARED = "shared"
    SEPARATE_MLP = "separate_mlp"
    SEPARATE_SPATIAL = "separate_spatial"


def _normalize_mode_str(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    m = str(mode).strip().lower()
    if m in ("", "none", "null"):
        return None
    aliases = {
        "shared": RatioHeadMode.SHARED.value,
        "default": RatioHeadMode.SHARED.value,
        "share": RatioHeadMode.SHARED.value,
        "coupled": RatioHeadMode.SHARED.value,
        # Separate MLP trunk (shared spatial trunk)
        "separate_mlp": RatioHeadMode.SEPARATE_MLP.value,
        "separate-mlp": RatioHeadMode.SEPARATE_MLP.value,
        "mlp": RatioHeadMode.SEPARATE_MLP.value,
        "separate_ratio_head": RatioHeadMode.SEPARATE_MLP.value,
        "separate_ratio_mlp": RatioHeadMode.SEPARATE_MLP.value,
        # Fully separate spatial head
        "separate_spatial": RatioHeadMode.SEPARATE_SPATIAL.value,
        "separate-spatial": RatioHeadMode.SEPARATE_SPATIAL.value,
        "spatial": RatioHeadMode.SEPARATE_SPATIAL.value,
        "full": RatioHeadMode.SEPARATE_SPATIAL.value,
        "duplicate_spatial": RatioHeadMode.SEPARATE_SPATIAL.value,
        "copy_spatial": RatioHeadMode.SEPARATE_SPATIAL.value,
        "separate_ratio_spatial_head": RatioHeadMode.SEPARATE_SPATIAL.value,
    }
    if m not in aliases:
        raise ValueError(
            f"Invalid ratio_head_mode={mode!r}. Expected one of: "
            f"{RatioHeadMode.SHARED.value!r}, {RatioHeadMode.SEPARATE_MLP.value!r}, {RatioHeadMode.SEPARATE_SPATIAL.value!r}."
        )
    return aliases[m]


def flags_from_ratio_head_mode(mode: Optional[str]) -> Tuple[bool, bool]:
    """
    Returns:
        separate_ratio_head: bool (separate scalar MLP branch)
        separate_ratio_spatial_head: bool (duplicate pyramid/conv stack)
    """
    m = _normalize_mode_str(mode) or RatioHeadMode.SHARED.value
    separate_spatial = (m == RatioHeadMode.SEPARATE_SPATIAL.value)
    separate_mlp = m in (RatioHeadMode.SEPARATE_MLP.value, RatioHeadMode.SEPARATE_SPATIAL.value)
    return separate_mlp, separate_spatial


def ratio_head_mode_from_flags(
    *,
    separate_ratio_head: bool,
    separate_ratio_spatial_head: bool,
) -> str:
    if bool(separate_ratio_spatial_head):
        return RatioHeadMode.SEPARATE_SPATIAL.value
    if bool(separate_ratio_head):
        return RatioHeadMode.SEPARATE_MLP.value
    return RatioHeadMode.SHARED.value


def resolve_ratio_head_mode(
    ratio_head_mode: Optional[str],
    *,
    separate_ratio_head: Optional[bool] = None,
    separate_ratio_spatial_head: Optional[bool] = None,
) -> str:
    """
    Resolve the effective ratio head mode from either:
      - the new enum field `ratio_head_mode`, OR
      - the legacy boolean flags (`separate_ratio_head`, `separate_ratio_spatial_head`).

    Precedence:
      1) ratio_head_mode when provided and non-empty
      2) separate_ratio_spatial_head
      3) separate_ratio_head
      4) shared
    """
    m = _normalize_mode_str(ratio_head_mode)
    if m is not None:
        return m
    return ratio_head_mode_from_flags(
        separate_ratio_head=bool(separate_ratio_head) if separate_ratio_head is not None else False,
        separate_ratio_spatial_head=bool(separate_ratio_spatial_head)
        if separate_ratio_spatial_head is not None
        else False,
    )

