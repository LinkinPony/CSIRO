from __future__ import annotations

"""
Training/validation step logic split into focused mixins.

This package replaces the previous monolithic `src/models/regressor/steps.py`.
`BiomassRegressor` continues to import:

    from .steps import RegressorStepsMixin
"""

from .epoch_metrics import EpochMetricsMixin
from .hooks import StepHooksMixin
from .losses import LossesMixin
from .predict import PredictionMixin
from .shared_step import SharedStepMixin
from .uw import UncertaintyWeightingMixin


class RegressorStepsMixin(
    UncertaintyWeightingMixin,
    PredictionMixin,
    LossesMixin,
    SharedStepMixin,
    StepHooksMixin,
    EpochMetricsMixin,
):
    """
    Composite mixin providing:
      - _shared_step (train/val)
      - training_step / validation_step
      - validation epoch metrics (val_r2)
      - uncertainty-weighting helpers
    """


__all__ = ["RegressorStepsMixin"]


