from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from ...layer_utils import average_layerwise_predictions


class LossesMixin:
    def _loss_ndvi_only(
        self,
        *,
        stage: str,
        head_type: str,
        z: Tensor,
        z_layers: Optional[List[Tensor]],
        ndvi_pred_from_head: Optional[Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        NDVI-only batch (no reg3 supervision). Optimize NDVI head only.
        """
        if not self.enable_ndvi:
            zero = (z.sum() * 0.0)
            self.log(f"{stage}_loss_ndvi", zero, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_loss", zero, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": zero}

        y_ndvi_only: Tensor = batch["y_ndvi"]  # (B,1)
        if head_type in ("fpn", "dpt"):
            if ndvi_pred_from_head is None:
                zero = (z.sum() * 0.0)
                self.log(f"{stage}_loss_ndvi", zero, on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"{stage}_loss", zero, on_step=False, on_epoch=True, prog_bar=True)
                return {"loss": zero}
            pred_ndvi_only = ndvi_pred_from_head
        else:
            if self.ndvi_head is None and (not self.use_layerwise_heads or self.layer_ndvi_heads is None):
                zero = (z.sum() * 0.0)
                self.log(f"{stage}_loss_ndvi", zero, on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"{stage}_loss", zero, on_step=False, on_epoch=True, prog_bar=True)
                return {"loss": zero}
            if self.use_layerwise_heads and self.layer_ndvi_heads is not None and z_layers is not None:
                preds_layers_ndvi = [head(z_layers[idx]) for idx, head in enumerate(self.layer_ndvi_heads)]
                pred_ndvi_only = average_layerwise_predictions(preds_layers_ndvi)
            else:
                pred_ndvi_only = self.ndvi_head(z)  # type: ignore[operator]

        loss_ndvi_only = F.mse_loss(pred_ndvi_only, y_ndvi_only)
        self.log(f"{stage}_loss_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse_ndvi", loss_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
        mae_ndvi_only = F.l1_loss(pred_ndvi_only, y_ndvi_only)
        self.log(f"{stage}_mae_ndvi", mae_ndvi_only, on_step=False, on_epoch=True, prog_bar=False)
        total_ndvi = self._uw_sum([("ndvi", loss_ndvi_only)])
        self.log(f"{stage}_loss", total_ndvi, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": total_ndvi}

    def _loss_supervised(
        self,
        *,
        stage: str,
        head_type: str,
        pred_reg3: Tensor,
        z: Tensor,
        z_layers: Optional[List[Tensor]],
        ratio_logits_pred: Optional[Tensor],
        ndvi_pred_from_head: Optional[Tensor],
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Full supervised batch (reg3 + optional ratio/5D + optional aux tasks).
        """
        # y_reg3 provided by dataset is already in normalized domain (z-score on g/m^2 when enabled)
        y_reg3: Tensor = batch["y_reg3"]  # (B, num_outputs)
        reg3_mask: Optional[Tensor] = batch.get("reg3_mask", None)
        y_height: Tensor = batch["y_height"]  # (B,1)
        y_ndvi: Tensor = batch["y_ndvi"]  # (B,1)
        y_species: Tensor = batch["y_species"]  # (B,)
        y_state: Tensor = batch["y_state"]  # (B,)

        if self.out_softplus is not None:
            pred_reg3 = self.out_softplus(pred_reg3)

        # Optional per-dimension supervision mask for reg3 (to support partial targets)
        if reg3_mask is not None:
            mask = reg3_mask.to(device=y_reg3.device, dtype=y_reg3.dtype)
        else:
            mask = torch.ones_like(y_reg3, dtype=y_reg3.dtype, device=y_reg3.device)
        diff_reg3 = pred_reg3 - y_reg3
        diff2_reg3 = (diff_reg3 * diff_reg3) * mask
        mask_sum_reg3 = mask.sum().clamp_min(1.0)
        loss_reg3_mse = diff2_reg3.sum() / mask_sum_reg3

        # Always log base reg3 metrics
        self.log(f"{stage}_loss_reg3_mse", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse_reg3", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)
        mae_reg3 = (diff_reg3.abs() * mask).sum() / mask_sum_reg3
        self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            per_dim_den = mask.sum(dim=0).clamp_min(1.0)
            per_dim_mse = diff2_reg3.sum(dim=0) / per_dim_den
            for i in range(per_dim_mse.shape[0]):
                self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)

        # --- Ratio MSE loss ---
        loss_ratio_mse: Optional[Tensor] = None
        if self.enable_ratio_head:
            y_ratio: Optional[Tensor] = batch.get("y_ratio", None)  # (B,3)
            ratio_mask: Optional[Tensor] = batch.get("ratio_mask", None)  # (B,1)
            if y_ratio is not None and ratio_mask is not None:
                if head_type in ("fpn", "dpt"):
                    ratio_logits = ratio_logits_pred
                elif self.use_layerwise_heads and self.layer_ratio_heads is not None and z_layers is not None:
                    logits_per_layer: List[Tensor] = []
                    for idx, head in enumerate(self.layer_ratio_heads):
                        logits_per_layer.append(head(z_layers[idx]))
                    ratio_logits = average_layerwise_predictions(logits_per_layer)
                else:
                    ratio_logits = self.ratio_head(z)  # type: ignore[operator]
                if ratio_logits is not None:
                    p_pred = F.softmax(ratio_logits, dim=-1)
                    p_true = y_ratio.clamp_min(0.0)
                    denom = p_true.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    p_true = p_true / denom
                    diff_ratio = p_pred - p_true
                    mse_per_sample = (diff_ratio * diff_ratio).sum(dim=-1, keepdim=True)  # (B,1)
                    m = ratio_mask.to(device=mse_per_sample.device, dtype=mse_per_sample.dtype)
                    num = (mse_per_sample * m).sum()
                    den = m.sum().clamp_min(1.0)
                    loss_ratio_mse = (num / den)
                    self.log(
                        f"{stage}_loss_ratio_mse",
                        loss_ratio_mse,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                    )

        # --- 5D weighted MSE loss ---
        loss_5d: Optional[Tensor] = None
        y_5d_g: Optional[Tensor] = batch.get("y_biomass_5d_g", None)  # (B,5) grams
        mask_5d: Optional[Tensor] = batch.get("biomass_5d_mask", None)  # (B,5)
        metrics_preds_5d: Optional[Tensor] = None
        metrics_targets_5d: Optional[Tensor] = None
        if self.enable_5d_loss and y_5d_g is not None and mask_5d is not None and self.enable_ratio_head:
            pred_total_gm2 = self._invert_reg3_to_g_per_m2(pred_reg3)  # (B,1)
            if head_type in ("fpn", "dpt"):
                ratio_logits = ratio_logits_pred
            elif self.use_layerwise_heads and self.layer_ratio_heads is not None and z_layers is not None:
                logits_per_layer_5d: List[Tensor] = []
                for idx, head in enumerate(self.layer_ratio_heads):
                    logits_per_layer_5d.append(head(z_layers[idx]))
                ratio_logits = average_layerwise_predictions(logits_per_layer_5d)
            else:
                ratio_logits = self.ratio_head(z)  # type: ignore[operator]

            if ratio_logits is None:
                p_pred = None
            else:
                p_pred = F.softmax(ratio_logits, dim=-1)  # (B,3)

            if p_pred is None:
                comp_gm2 = torch.zeros((pred_total_gm2.size(0), 3), device=pred_total_gm2.device, dtype=pred_total_gm2.dtype)
            else:
                comp_gm2 = p_pred * pred_total_gm2  # (B,3)

            clover_pred = comp_gm2[:, 0]
            dead_pred = comp_gm2[:, 1]
            green_pred = comp_gm2[:, 2]
            gdm_pred = clover_pred + green_pred
            total_pred = pred_total_gm2.squeeze(-1)
            pred_5d_gm2 = torch.stack([clover_pred, dead_pred, green_pred, gdm_pred, total_pred], dim=-1)

            metrics_preds_5d = pred_5d_gm2 * float(self._area_m2)
            metrics_targets_5d = y_5d_g.to(device=metrics_preds_5d.device, dtype=metrics_preds_5d.dtype)

            y_5d_gm2 = y_5d_g.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype) / float(self._area_m2)

            pred_5d_input = pred_5d_gm2
            target_5d_input = y_5d_gm2
            if self.log_scale_targets:
                pred_5d_input = torch.log1p(torch.clamp(pred_5d_input, min=0.0))
                target_5d_input = torch.log1p(torch.clamp(target_5d_input, min=0.0))

            if self._use_biomass_5d_zscore:
                mean_5d = self._biomass_5d_mean.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype)  # type: ignore[union-attr]
                std_5d = torch.clamp(
                    self._biomass_5d_std.to(device=pred_5d_gm2.device, dtype=pred_5d_gm2.dtype),  # type: ignore[union-attr]
                    min=1e-8,
                )
                pred_5d = (pred_5d_input - mean_5d) / std_5d
                target_5d = (target_5d_input - mean_5d) / std_5d
            else:
                pred_5d = pred_5d_input
                target_5d = target_5d_input

            w = self.biomass_5d_weights.to(device=pred_5d.device, dtype=pred_5d.dtype)  # (5,)
            m5 = mask_5d.to(device=pred_5d.device, dtype=pred_5d.dtype)
            diff_5d = (pred_5d - target_5d) * m5
            diff2_5d = diff_5d * diff_5d
            per_dim_den = m5.sum(dim=0).clamp_min(1.0)
            mse_per_dim = diff2_5d.sum(dim=0) / per_dim_den  # (5,)
            valid_weight = w * (per_dim_den > 0).to(dtype=w.dtype)
            total_w = valid_weight.sum().clamp_min(1e-8)
            loss_5d = (w * mse_per_dim).sum() / total_w

            names_5d = ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g", "GDM_g", "Dry_Total_g"]
            for i, name in enumerate(names_5d):
                self.log(f"{stage}_mse_5d_{name}", mse_per_dim[i], on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_loss_5d_weighted", loss_5d, on_step=False, on_epoch=True, prog_bar=False)

        # Aggregate reg3-related losses for logging
        loss_reg3_total = loss_reg3_mse
        if loss_ratio_mse is not None:
            loss_reg3_total = loss_reg3_total + loss_ratio_mse
        if loss_5d is not None:
            loss_reg3_total = loss_reg3_total + loss_5d
        self.log(f"{stage}_loss_reg3", loss_reg3_total, on_step=False, on_epoch=True, prog_bar=False)

        # If MTL is disabled or all auxiliary tasks are off, optimize only the reg3 path.
        if (not self.mtl_enabled) or (
            self.enable_height is False
            and self.enable_ndvi is False
            and self.enable_species is False
            and self.enable_state is False
        ):
            if self.loss_weighting == "uw":
                named_losses_simple: List[Tuple[str, Tensor]] = [("reg3", loss_reg3_mse)]
                if loss_ratio_mse is not None:
                    named_losses_simple.append(("ratio", loss_ratio_mse))
                if loss_5d is not None:
                    named_losses_simple.append(("biomass_5d", loss_5d))
                total_loss = self._uw_sum(named_losses_simple)
            else:
                total_loss = loss_reg3_total
            self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_mae", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse", loss_reg3_mse, on_step=False, on_epoch=True, prog_bar=False)

            if metrics_preds_5d is not None and metrics_targets_5d is not None:
                preds_out = metrics_preds_5d.detach()
                targets_out = metrics_targets_5d.detach()
            else:
                preds_out = self._invert_reg3_to_grams(pred_reg3.detach())
                targets_out = batch.get("y_reg3_g", None)
                if targets_out is None:
                    y_gm2 = batch.get("y_reg3_g_m2", None)
                    if y_gm2 is not None:
                        targets_out = y_gm2 * float(self._area_m2)
                    else:
                        targets_out = self._invert_reg3_to_grams(y_reg3.detach())
            return {
                "loss": total_loss,
                "mae": mae_reg3,
                "mse": loss_reg3_mse,
                "preds": preds_out,
                "targets": targets_out,
            }

        # Otherwise, compute enabled auxiliary task heads and losses
        pred_height = None
        pred_ndvi = None
        logits_species = None
        logits_state = None
        if head_type in ("fpn", "dpt"):
            pred_height = self.height_head(z) if self.enable_height else None  # type: ignore[assignment]
            pred_ndvi = ndvi_pred_from_head if self.enable_ndvi else None
            logits_species = self.species_head(z) if self.enable_species else None  # type: ignore[assignment]
            logits_state = self.state_head(z) if self.enable_state else None  # type: ignore[assignment]
        elif self.use_layerwise_heads and z_layers is not None:
            if self.enable_height and self.layer_height_heads is not None:
                height_preds_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_height_heads):
                    height_preds_layers.append(head(z_layers[idx]))
                pred_height = average_layerwise_predictions(height_preds_layers)
            if self.enable_ndvi and self.layer_ndvi_heads is not None:
                ndvi_preds_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_ndvi_heads):
                    ndvi_preds_layers.append(head(z_layers[idx]))
                pred_ndvi = average_layerwise_predictions(ndvi_preds_layers)
            if self.enable_species and self.layer_species_heads is not None:
                species_logits_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_species_heads):
                    species_logits_layers.append(head(z_layers[idx]))
                logits_species = average_layerwise_predictions(species_logits_layers)
            if self.enable_state and self.layer_state_heads is not None:
                state_logits_layers: List[Tensor] = []
                for idx, head in enumerate(self.layer_state_heads):
                    state_logits_layers.append(head(z_layers[idx]))
                logits_state = average_layerwise_predictions(state_logits_layers)
        else:
            pred_height = self.height_head(z) if self.enable_height else None  # type: ignore[assignment]
            pred_ndvi = self.ndvi_head(z) if self.enable_ndvi else None  # type: ignore[assignment]
            logits_species = self.species_head(z) if self.enable_species else None  # type: ignore[assignment]
            logits_state = self.state_head(z) if self.enable_state else None  # type: ignore[assignment]

        named_losses: List[Tuple[str, Tensor]] = []
        named_losses.append(("reg3", loss_reg3_mse))
        if loss_ratio_mse is not None:
            named_losses.append(("ratio", loss_ratio_mse))
        if loss_5d is not None:
            named_losses.append(("biomass_5d", loss_5d))

        if self.enable_height:
            loss_height = F.mse_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_loss_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
            mae_height = F.l1_loss(pred_height, y_height)  # type: ignore[arg-type]
            self.log(f"{stage}_mae_height", mae_height, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("height", loss_height))

        if self.enable_ndvi:
            ndvi_mask: Optional[Tensor] = batch.get("ndvi_mask", None)
            if ndvi_mask is not None:
                m_nd = ndvi_mask.to(device=y_ndvi.device, dtype=y_ndvi.dtype)  # type: ignore[arg-type]
            else:
                m_nd = torch.ones_like(y_ndvi, dtype=y_ndvi.dtype, device=y_ndvi.device)  # type: ignore[arg-type]

            if pred_ndvi is None:
                zero_nd = (z.sum() * 0.0)
                loss_ndvi = zero_nd
                mae_ndvi = zero_nd
            else:
                diff_ndvi = pred_ndvi - y_ndvi  # type: ignore[operator]
                diff2_ndvi = (diff_ndvi * diff_ndvi) * m_nd
                mask_sum_ndvi = m_nd.sum().clamp_min(0.0)
                if mask_sum_ndvi > 0:
                    loss_ndvi = diff2_ndvi.sum() / mask_sum_ndvi
                    mae_ndvi = (diff_ndvi.abs() * m_nd).sum() / mask_sum_ndvi
                else:
                    zero_nd = diff2_ndvi.sum() * 0.0
                    loss_ndvi = zero_nd
                    mae_ndvi = zero_nd
            self.log(f"{stage}_loss_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mse_ndvi", loss_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f"{stage}_mae_ndvi", mae_ndvi, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("ndvi", loss_ndvi))

        if self.enable_species and logits_species is not None:
            loss_species = F.cross_entropy(logits_species, y_species)
            self.log(f"{stage}_loss_species", loss_species, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_species = (logits_species.argmax(dim=-1) == y_species).float().mean()
            self.log(f"{stage}_acc_species", acc_species, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("species", loss_species))

        if self.enable_state and logits_state is not None:
            loss_state = F.cross_entropy(logits_state, y_state)
            self.log(f"{stage}_loss_state", loss_state, on_step=False, on_epoch=True, prog_bar=False)
            with torch.no_grad():
                acc_state = (logits_state.argmax(dim=-1) == y_state).float().mean()
            self.log(f"{stage}_acc_state", acc_state, on_step=False, on_epoch=True, prog_bar=False)
            named_losses.append(("state", loss_state))

        total_loss = self._uw_sum(named_losses)

        mae = F.l1_loss(pred_reg3, y_reg3)
        mse = F.mse_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)

        if metrics_preds_5d is not None and metrics_targets_5d is not None:
            preds_out = metrics_preds_5d.detach()
            targets_out = metrics_targets_5d.detach()
        else:
            preds_out = self._invert_reg3_to_grams(pred_reg3.detach())
            targets_out = batch.get("y_reg3_g", None)
            if targets_out is None:
                y_gm2 = batch.get("y_reg3_g_m2", None)
                if y_gm2 is not None:
                    targets_out = y_gm2 * float(self._area_m2)
                else:
                    targets_out = self._invert_reg3_to_grams(y_reg3.detach())

        return {
            "loss": total_loss,
            "mae": mae,
            "mse": mse,
            "preds": preds_out,
            "targets": targets_out,
        }


