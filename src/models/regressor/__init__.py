"""
Regressor package.

This used to live in a single large `src/models/regressor.py` module. It has been
split into a proper package to keep the codebase maintainable while preserving
the public import path:

    from src.models.regressor import BiomassRegressor
"""

from __future__ import annotations

from importlib.util import find_spec

# NOTE:
# -----
# This package contains both:
#   - lightweight torch modules (heads) used for inference, and
#   - training helpers that depend on PyTorch Lightning.
#
# Kaggle (and other minimal inference environments) may not ship with Lightning
# installed. Importing training classes unconditionally here would break *any*
# import of `src.models.regressor.*` (including inference-only heads) because
# Python evaluates this __init__.py first when importing submodules.
#
# To keep inference lightweight and robust, only expose Lightning-dependent
# symbols when the dependency is available.

try:
    _HAS_LIGHTNING = find_spec("lightning") is not None
except Exception:
    _HAS_LIGHTNING = False

if _HAS_LIGHTNING:
    # Be defensive: some environments may have a partially-installed `lightning`
    # package (or version mismatches) that still fails at import-time.
    try:
        from .biomass_regressor import BiomassRegressor  # noqa: F401
        from .manifold_mixup import ManifoldMixup  # noqa: F401
    except Exception:
        _HAS_LIGHTNING = False

__all__ = ["BiomassRegressor", "ManifoldMixup"] if _HAS_LIGHTNING else []


