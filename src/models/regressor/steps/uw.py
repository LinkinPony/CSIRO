from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor


class UncertaintyWeightingMixin:
    def _uw_sum(self, named_losses: List[Tuple[str, Tensor]]) -> Tensor:
        if (
            self.loss_weighting != "uw"
            or self._uw_task_params is None
            or len(named_losses) == 0
        ):
            return torch.stack([loss for _, loss in named_losses]).mean()
        total = 0.0
        for name, loss in named_losses:
            s = self._uw_task_params.get(name, None)
            if s is None:
                # If this task wasn't registered at init, fall back to equal weighting
                total = total + loss
            else:
                total = total + 0.5 * (torch.exp(-s) * loss + s)
        # Normalize by number of tasks for scale stability
        return total if len(named_losses) == 1 else (total / len(named_losses))

    def _log_uw_parameters(self, stage: str) -> None:
        """
        Log UW parameters per task:
          - {stage}_uw_logvar_{task}: s (log variance)
          - {stage}_uw_sigma_{task}: sigma = exp(0.5 * s)
        """
        if self.loss_weighting != "uw" or self._uw_task_params is None:
            return
        for name, s_param in self._uw_task_params.items():
            try:
                s = s_param.detach()
            except Exception:
                s = s_param
            sigma = torch.exp(0.5 * s)
            # Ensure scalar logging
            self.log(
                f"{stage}_uw_logvar_{name}",
                s.squeeze(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{stage}_uw_sigma_{name}",
                sigma.squeeze(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )


