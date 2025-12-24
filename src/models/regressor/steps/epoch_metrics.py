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

        # --- R^2 in log-space ---
        # IMPORTANT:
        # - R^2 should use the mean of the evaluation targets (here: validation epoch targets)
        #   as the baseline. Using a train/global mean can inflate/deflate the value.
        # - When `log_scale_targets` is enabled, z-score means are in log-space already; treating
        #   them as linear g/m^2 and then applying log1p again would double-transform and
        #   severely inflate SS_tot (and hence R^2). We avoid this by defaulting to val mean.
        mean_log_val = torch.mean(targets_log, dim=0)

        ss_res_per = torch.sum((targets_log - preds_log) ** 2, dim=0)
        ss_tot_per = torch.sum((targets_log - mean_log_val) ** 2, dim=0)
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

        # Standard validation R^2 (baseline = val mean in log-space)
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True)

        # Optional: also log a "global baseline" variant for debugging/comparison.
        # This uses stored train means when available, but correctly handles log_scale_targets.
        mean_log_global = mean_log_val
        try:
            if preds_log.shape[1] == 5 and getattr(self, "_biomass_5d_mean", None) is not None:
                mean_gm2 = self._biomass_5d_mean.to(  # type: ignore[union-attr]
                    device=targets_log.device, dtype=targets_log.dtype
                )
                if bool(getattr(self, "log_scale_targets", False)):
                    mean_gm2 = torch.expm1(mean_gm2).clamp_min(0.0)
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log_global = torch.log1p(mean_g.clamp_min(0.0))
            elif getattr(self, "_reg3_mean", None) is not None:
                mean_gm2 = self._reg3_mean.to(  # type: ignore[union-attr]
                    device=targets_log.device, dtype=targets_log.dtype
                )
                if bool(getattr(self, "log_scale_targets", False)):
                    mean_gm2 = torch.expm1(mean_gm2).clamp_min(0.0)
                mean_g = mean_gm2 * float(self._area_m2)
                mean_log_global = torch.log1p(mean_g.clamp_min(0.0))
        except Exception:
            mean_log_global = mean_log_val

        ss_tot_per_g = torch.sum((targets_log - mean_log_global) ** 2, dim=0)
        r2_per_g = 1.0 - (ss_res_per / (ss_tot_per_g + eps))
        if preds_log.shape[1] == 5:
            valid_g = torch.isfinite(r2_per_g)
            w_eff_g = weights * valid_g.to(dtype=weights.dtype)
            denom_g = w_eff_g.sum().clamp_min(eps)
            r2_g = (w_eff_g * r2_per_g).sum() / denom_g
        else:
            r2_g = torch.mean(r2_per_g)
        self.log("val_r2_global", r2_g, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self) -> None:
        self._log_uw_parameters("train")


