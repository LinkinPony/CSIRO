from __future__ import annotations

import torch


class EpochMetricsMixin:
    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
        # Always log UW parameters for validation epoch
        self._log_uw_parameters("val")
        if len(self._val_preds) == 0:
            return
        preds = torch.cat(self._val_preds, dim=0)  # grams, shape (N, D)
        targets = torch.cat(self._val_targets, dim=0)  # grams, shape (N, D)

        eps = 1e-8
        preds_clamp = preds.clamp_min(0.0)
        targets_clamp = targets.clamp_min(0.0)
        preds_log = torch.log1p(preds_clamp)
        targets_log = torch.log1p(targets_clamp)

        # Baseline mean per dimension in log-space
        if preds_log.shape[1] == 5 and self._biomass_5d_mean is not None:
            try:
                mean_gm2 = self._biomass_5d_mean.to(  # type: ignore[union-attr]
                    device=targets_log.device, dtype=targets_log.dtype
                )
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log = torch.log1p(mean_g.clamp_min(0.0))
            except Exception:
                mean_log = torch.mean(targets_log, dim=0)
        elif self._reg3_mean is not None:
            try:
                mean_gm2 = self._reg3_mean.to(  # type: ignore[union-attr]
                    device=targets_log.device, dtype=targets_log.dtype
                )
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log = torch.log1p(mean_g.clamp_min(0.0))
            except Exception:
                mean_log = torch.mean(targets_log, dim=0)
        else:
            mean_log = torch.mean(targets_log, dim=0)

        ss_res_per = torch.sum((targets_log - preds_log) ** 2, dim=0)
        ss_tot_per = torch.sum((targets_log - mean_log) ** 2, dim=0)
        r2_per = 1.0 - (ss_res_per / (ss_tot_per + eps))

        if preds_log.shape[1] == 5:
            weights = torch.tensor(
                [0.1, 0.1, 0.1, 0.2, 0.5],
                device=r2_per.device,
                dtype=r2_per.dtype,
            )
            valid = torch.isfinite(r2_per)
            w_eff = weights * valid.to(dtype=weights.dtype)
            denom = w_eff.sum().clamp_min(eps)
            r2 = (w_eff * r2_per).sum() / denom
        else:
            r2 = torch.mean(r2_per)

        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        self._log_uw_parameters("train")


