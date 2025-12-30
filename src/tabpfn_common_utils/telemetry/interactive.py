"""
Interactive telemetry hooks used by TabPFN.

Upstream uses these to acknowledge an anonymous session. For offline inference
we intentionally do nothing.
"""

from __future__ import annotations

from typing import Any


def ping(*args: Any, **kwargs: Any) -> None:
    """No-op."""


def capture_session(*args: Any, **kwargs: Any) -> None:
    """No-op."""


__all__ = ["ping", "capture_session"]


