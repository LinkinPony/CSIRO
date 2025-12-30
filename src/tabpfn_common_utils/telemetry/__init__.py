"""
Minimal telemetry API used by TabPFN.

TabPFN decorates a few estimator methods with `track_model_call(...)` and reports
model metadata via `set_model_config(...)`. For offline inference in this repo
we do not emit any telemetry, so these are implemented as no-ops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def track_model_call(*args: Any, **kwargs: Any):
    """
    No-op decorator used by TabPFN for telemetry.

    Supports both:
      - `@track_model_call(...)`
      - `@track_model_call`  (rare, but supported for completeness)
    """

    # Case: used directly as `@track_model_call` (no args)
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        return func

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        return func

    return decorator


def set_model_config(*args: Any, **kwargs: Any) -> None:
    """No-op hook used by TabPFN to report model configuration for telemetry."""


__all__ = ["track_model_call", "set_model_config"]


