"""
Regressor package.

This used to live in a single large `src/models/regressor.py` module. It has been
split into a proper package to keep the codebase maintainable while preserving
the public import path:

    from src.models.regressor import BiomassRegressor
"""

from .biomass_regressor import BiomassRegressor
from .manifold_mixup import ManifoldMixup

__all__ = ["BiomassRegressor", "ManifoldMixup"]


