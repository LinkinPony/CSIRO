from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from PIL import Image


class SharedStepMixin:
    """
    Orchestrates batch selection, augmentation decision, prediction, and loss computation.
    """

    def _maybe_dump_model_inputs(
        self,
        *,
        images: Tensor,
        batch: Any,
        stage: str,
        use_cutmix: bool,
        is_ndvi_only: bool,
    ) -> None:
        """
        Optional debug utility: write out the *final* model input images (after CutMix)
        for quick visual sanity checks.
        """
        cfg = dict(getattr(self, "_debug_input_dump_cfg", {}) or {})
        if not bool(cfg.get("enabled", False)):
            return

        # Avoid multi-GPU / multi-process spam.
        try:
            tr = getattr(self, "trainer", None)
            if tr is not None and int(getattr(tr, "global_rank", 0)) != 0:
                return
        except Exception:
            pass

        stage_l = str(stage or "").strip().lower()
        stages_cfg = cfg.get("stages", cfg.get("stage", "train"))
        if isinstance(stages_cfg, str):
            allowed = {stages_cfg.strip().lower()}
        else:
            try:
                allowed = {str(x).strip().lower() for x in list(stages_cfg)}
            except Exception:
                allowed = {"train"}
        if stage_l and stage_l not in allowed:
            return

        if is_ndvi_only and (not bool(cfg.get("include_ndvi_only", False))):
            return

        # Limit dumping frequency / amount.
        step = int(getattr(self, "global_step", 0) or 0)
        every_n = int(cfg.get("every_n_steps", cfg.get("every_n_train_steps", 1)) or 0)
        if every_n > 0 and (step % every_n) != 0:
            return

        # Avoid dumping multiple times for the same (stage, global_step) e.g. with grad accumulation.
        last_key = getattr(self, "_debug_input_dump_last_key", None)
        key = (stage_l, step)
        if last_key == key:
            return
        setattr(self, "_debug_input_dump_last_key", key)

        written = int(getattr(self, "_debug_input_dump_written", 0) or 0)
        max_images = int(cfg.get("max_images", 16) or 0)
        if max_images > 0 and written >= max_images:
            return

        n_per = int(cfg.get("num_images_per_step", cfg.get("n_images_per_step", 4)) or 1)
        if n_per <= 0:
            return

        # Resolve output directory.
        out_dir_cfg = cfg.get("out_dir", cfg.get("dir", None))
        if out_dir_cfg:
            out_dir = Path(str(out_dir_cfg))
            if not out_dir.is_absolute():
                base = getattr(self, "_run_log_dir", None)
                out_dir = (Path(str(base)) / out_dir) if base else out_dir
        else:
            base = getattr(self, "_run_log_dir", None)
            out_dir = (Path(str(base)) / "debug" / "inputs") if base else (Path("outputs") / "debug" / "inputs")
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        # Determine file format.
        fmt = str(cfg.get("format", "png") or "png").strip().lower()
        if fmt == "jpg":
            fmt = "jpeg"
        if fmt not in {"png", "jpeg", "webp"}:
            fmt = "png"

        denorm = bool(cfg.get("denormalize", True))
        mean = getattr(self, "_input_image_mean", None)
        std = getattr(self, "_input_image_std", None)
        if denorm and (mean is None or std is None):
            denorm = False

        # Resolve per-sample ids (best-effort).
        ids: Optional[List[str]] = None
        try:
            if isinstance(batch, dict) and "image_id" in batch:
                raw = batch.get("image_id")
                if isinstance(raw, (list, tuple)):
                    ids = [str(x) for x in list(raw)]
                elif raw is not None:
                    # Single value (unlikely with collate) -> broadcast
                    ids = [str(raw)]
        except Exception:
            ids = None

        cutmix_perm = None
        try:
            if use_cutmix and isinstance(batch, dict) and "_cutmix_perm" in batch:
                p = batch.get("_cutmix_perm")
                if isinstance(p, torch.Tensor):
                    cutmix_perm = p.detach().cpu().tolist()
        except Exception:
            cutmix_perm = None

        def _sanitize_id(s: str) -> str:
            # Keep filenames simple/safe.
            s = str(s or "").strip()
            if not s:
                return "unknown"
            out = []
            for ch in s:
                if ch.isalnum() or ch in ("-", "_"):
                    out.append(ch)
                else:
                    out.append("_")
            s2 = "".join(out)
            # avoid extremely long filenames
            return s2[:80] if len(s2) > 80 else s2

        # Best-effort dump; never crash training.
        try:
            x = images.detach().float()
            if x.ndim != 4:
                return
            b = int(x.shape[0])
            if b <= 0:
                return
            x = x.cpu()
            n_take = min(b, n_per)
            if max_images > 0:
                n_take = min(n_take, max_images - written)
            if n_take <= 0:
                return

            for i in range(n_take):
                t = x[i]
                if denorm:
                    # mean/std are expected shape (C,1,1)
                    t = t * std + mean
                t = t.clamp(0.0, 1.0)

                # Tensor (C,H,W) -> uint8 HWC
                arr = (
                    (t * 255.0)
                    .add(0.5)
                    .clamp(0.0, 255.0)
                    .to(dtype=torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .numpy()
                )
                im = Image.fromarray(arr)

                # Include original id(s) in filenames for traceability.
                id_a = None
                id_b = None
                if ids is not None:
                    if len(ids) == b:
                        id_a = ids[i]
                    elif len(ids) == 1:
                        id_a = ids[0]
                if cutmix_perm is not None and ids is not None and len(ids) == b:
                    try:
                        j = int(cutmix_perm[i])
                        if 0 <= j < b:
                            id_b = ids[j]
                    except Exception:
                        id_b = None

                id_tag = _sanitize_id(id_a) if id_a is not None else "unknown"
                if id_b is not None:
                    id_tag = f"{id_tag}__mix_{_sanitize_id(id_b)}"

                name = f"{stage_l}_step{step:06d}_{id_tag}_img{written + i:05d}"
                if use_cutmix:
                    name += "_cutmix"
                if is_ndvi_only:
                    name += "_ndvi"
                path = out_dir / f"{name}.{fmt}"

                save_kwargs = {}
                if fmt == "jpeg":
                    save_kwargs = {
                        "quality": int(cfg.get("jpeg_quality", 95)),
                        "optimize": True,
                    }
                elif fmt == "png":
                    # 0 (none) .. 9 (max); default 6
                    save_kwargs = {"compress_level": int(cfg.get("png_compress_level", 6))}
                elif fmt == "webp":
                    save_kwargs = {"quality": int(cfg.get("webp_quality", 90))}

                # Ensure parent exists (e.g. if out_dir is a file path by mistake, mkdir above would fail).
                try:
                    os.makedirs(str(path.parent), exist_ok=True)
                except Exception:
                    pass
                im.save(str(path), **save_kwargs)

            setattr(self, "_debug_input_dump_written", written + n_take)
        except Exception:
            return

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        # When multiple dataloaders are used, Lightning may deliver a list/tuple of batches.
        # For alternating training, process only ONE sub-batch per step to avoid holding multiple graphs.
        if isinstance(batch, (list, tuple)):
            flat_batches: List[Any] = []
            for sub in batch:
                if isinstance(sub, (list, tuple)):
                    for sb in sub:
                        flat_batches.append(sb)
                else:
                    flat_batches.append(sub)

            use_ndvi_only = False
            if stage == "train" and self.enable_ndvi_dense:
                try:
                    use_ndvi_only = bool(torch.rand(()) < self._ndvi_dense_prob)
                except Exception:
                    use_ndvi_only = False

            selected: Optional[Any] = None
            if use_ndvi_only:
                selected = next(
                    (x for x in flat_batches if isinstance(x, dict) and bool(x.get("ndvi_only", False))),
                    None,
                )
            else:
                selected = next(
                    (x for x in flat_batches if isinstance(x, dict) and not bool(x.get("ndvi_only", False))),
                    None,
                )
            if selected is None:
                selected = next((x for x in flat_batches if isinstance(x, dict)), flat_batches[0])
            return self._shared_step(selected, stage)

        # batch is a dict from the dataset
        is_ndvi_only: bool = bool(batch.get("ndvi_only", False))

        # Decide which augmentation (CutMix / manifold mixup) to apply for this batch.
        use_cutmix = False
        use_mixup = False
        if stage == "train" and (not is_ndvi_only):
            cutmix_enabled = self._cutmix_main is not None
            cutmix_prob = 0.0
            if cutmix_enabled:
                try:
                    cutmix_enabled = bool(getattr(self._cutmix_main, "cfg", None) and self._cutmix_main.cfg.enabled)
                    cutmix_prob = float(self._cutmix_main.cfg.prob)
                except Exception:
                    cutmix_enabled = False
                    cutmix_prob = 0.0

            mixup_enabled = self._manifold_mixup is not None and bool(self._manifold_mixup.enabled)
            mixup_prob = 0.0
            if mixup_enabled:
                try:
                    mixup_prob = float(self._manifold_mixup.prob)
                except Exception:
                    mixup_prob = 0.0

            cut_trigger = False
            mix_trigger = False
            if cutmix_enabled and cutmix_prob > 0.0:
                try:
                    cut_trigger = bool(torch.rand(()) < cutmix_prob)
                except Exception:
                    cut_trigger = False
            if mixup_enabled and mixup_prob > 0.0:
                try:
                    mix_trigger = bool(torch.rand(()) < mixup_prob)
                except Exception:
                    mix_trigger = False

            if cut_trigger and mix_trigger:
                try:
                    choose_cut = bool(torch.rand(()) < 0.5)
                except Exception:
                    choose_cut = True
                if choose_cut:
                    use_cutmix, use_mixup = True, False
                else:
                    use_cutmix, use_mixup = False, True
            elif cut_trigger:
                use_cutmix = True
            elif mix_trigger:
                use_mixup = True

        images_any = batch["image"]
        # In rare cases a dataset/transform may return multi-view images even for ndvi-only.
        # Ensure the NDVI-only path always sees a single tensor.
        if is_ndvi_only and isinstance(images_any, (list, tuple)) and len(images_any) >= 1:
            images_any = images_any[0]

        # Multi-view path (AugMix consistency): batch["image"] is a tuple/list of tensors:
        #   (clean, aug1, aug2)  or (clean, aug1)
        is_multiview = isinstance(images_any, (list, tuple)) and len(images_any) >= 2
        if is_multiview and (not is_ndvi_only) and str(stage).lower() == "train":
            views = list(images_any)
            images_clean = views[0]
            images_aug_list = views[1:]

            # For now we do NOT apply CutMix in multi-view mode (would need view-consistent CutMix).
            use_cutmix = False

            # Optional debug dump: save the clean view (final image fed to the model in the supervised branch).
            if isinstance(images_clean, torch.Tensor):
                self._maybe_dump_model_inputs(
                    images=images_clean,
                    batch=batch,
                    stage=stage,
                    use_cutmix=use_cutmix,
                    is_ndvi_only=is_ndvi_only,
                )

            # View-consistent manifold mixup: reuse the same (lam, perm) for all views.
            mix_lam = None
            mix_perm = None
            if use_mixup and getattr(self, "_manifold_mixup", None) is not None and isinstance(images_clean, torch.Tensor):
                try:
                    bsz = int(images_clean.size(0))
                except Exception:
                    bsz = 0
                if bsz >= 2:
                    try:
                        mix_lam, mix_perm = self._manifold_mixup.sample_params(bsz=bsz, device=images_clean.device)  # type: ignore[union-attr]
                    except Exception:
                        mix_lam, mix_perm = None, None
                else:
                    # Avoid cache-mode mixup when training multi-view (would update cache multiple times per step).
                    use_mixup = False

            head_type = str(getattr(self, "_head_type", "mlp")).lower()

            # 1) Supervised branch uses the clean view and mixes labels.
            (
                pred_reg3_c,
                z_c,
                z_layers_c,
                ratio_logits_pred_c,
                ndvi_pred_c,
                batch_mixed,
                pred_reg3_layers_c,
                ratio_logits_layers_c,
            ) = self._predict_reg3_and_z(
                images=images_clean,
                batch=batch,
                stage=stage,
                use_mixup=use_mixup,
                is_ndvi_only=is_ndvi_only,
                mixup_lam=mix_lam,
                mixup_perm=mix_perm,
                mixup_mix_labels=True,
            )

            # 2) Consistency branches: forward on aug views with the SAME mixup params but WITHOUT mixing labels.
            cons_cfg = dict(getattr(self, "_augmix_consistency_cfg", {}) or {})
            cons_enabled = bool(cons_cfg.get("enabled", True))
            if cons_enabled and images_aug_list:
                # We support 1 or 2 aug views (AugMix-style).
                preds_reg3: List[Tensor] = [pred_reg3_c]
                zs: List[Tensor] = [z_c]
                z_layers_list: List[Optional[List[Tensor]]] = [z_layers_c]
                ratio_logits_list: List[Optional[Tensor]] = [ratio_logits_pred_c]

                for img_v in images_aug_list[:2]:
                    (
                        pr,
                        zv,
                        zl,
                        rlog,
                        _nd,
                        _b2,
                        _pr_layers,
                        _rlog_layers,
                    ) = self._predict_reg3_and_z(
                        images=img_v,
                        batch=batch_mixed,
                        stage=stage,
                        use_mixup=use_mixup,
                        is_ndvi_only=is_ndvi_only,
                        mixup_lam=mix_lam,
                        mixup_perm=mix_perm,
                        mixup_mix_labels=False,
                    )
                    preds_reg3.append(pr)
                    zs.append(zv)
                    z_layers_list.append(zl)
                    ratio_logits_list.append(rlog)

                # --- reg3 consistency (MSE to mean, analogous to JSD mixture) ---
                try:
                    preds_for_cons = []
                    sp = getattr(self, "out_softplus", None)
                    for p in preds_reg3:
                        preds_for_cons.append(sp(p) if sp is not None else p)
                    pred_bar = torch.stack(preds_for_cons, dim=0).mean(dim=0)
                    loss_cons_reg3 = torch.zeros((), device=pred_bar.device, dtype=pred_bar.dtype)
                    for p in preds_for_cons:
                        loss_cons_reg3 = loss_cons_reg3 + F.mse_loss(p, pred_bar)
                    loss_cons_reg3 = loss_cons_reg3 / float(len(preds_for_cons))
                except Exception:
                    loss_cons_reg3 = pred_reg3_c.sum() * 0.0

                # --- ratio consistency (JSD on probabilities) ---
                loss_cons_ratio = pred_reg3_c.sum() * 0.0
                if bool(getattr(self, "enable_ratio_head", False)):
                    try:
                        probs: List[Tensor] = []

                        def _get_ratio_logits(zv: Tensor, zl: Optional[List[Tensor]], r_pred: Optional[Tensor]) -> Optional[Tensor]:
                            if isinstance(r_pred, torch.Tensor):
                                return r_pred
                            rh = getattr(self, "ratio_head", None)
                            if rh is None:
                                return None
                            # Layerwise heads (fallback path)
                            if bool(getattr(self, "use_layerwise_heads", False)) and zl is not None:
                                lrh = getattr(self, "layer_ratio_heads", None)
                                if isinstance(lrh, list) and len(lrh) > 0:
                                    logits_per_layer = [head(zl[idx]) for idx, head in enumerate(lrh)]
                                    from ...layer_utils import fuse_layerwise_predictions
                                    w = self._get_backbone_layer_fusion_weights(
                                        device=logits_per_layer[0].device, dtype=logits_per_layer[0].dtype
                                    )
                                    return fuse_layerwise_predictions(logits_per_layer, weights=w)
                            return rh(zv)  # type: ignore[operator]

                        for zv, zl, r_pred in zip(zs, z_layers_list, ratio_logits_list):
                            logits = _get_ratio_logits(zv, zl, r_pred)
                            if logits is None:
                                probs = []
                                break
                            probs.append(F.softmax(logits, dim=-1))

                        if probs:
                            eps = float(cons_cfg.get("eps", 1e-7))
                            n = float(len(probs))
                            p_mix = torch.clamp(sum(probs) / n, eps, 1.0).log()
                            jsd = torch.zeros((), device=p_mix.device, dtype=p_mix.dtype)
                            for p in probs:
                                jsd = jsd + F.kl_div(p_mix, p, reduction="batchmean")
                            loss_cons_ratio = jsd / n
                    except Exception:
                        loss_cons_ratio = pred_reg3_c.sum() * 0.0

                w_reg3 = float(cons_cfg.get("weight_reg3", 1.0))
                w_ratio = float(cons_cfg.get("weight_ratio_jsd", cons_cfg.get("weight_ratio", 12.0)))
                self.log(f"{stage}_loss_cons_reg3", loss_cons_reg3, on_step=False, on_epoch=True, prog_bar=False)
                self.log(f"{stage}_loss_cons_ratio_jsd", loss_cons_ratio, on_step=False, on_epoch=True, prog_bar=False)

                loss_cons_total = (w_reg3 * loss_cons_reg3) + (w_ratio * loss_cons_ratio)
                self.log(f"{stage}_loss_consistency", loss_cons_total, on_step=False, on_epoch=True, prog_bar=False)

                # If Uncertainty Weighting is enabled, add consistency as its own UW task.
                # Otherwise, add the (weighted) consistency penalty directly.
                use_uw_task = bool(cons_cfg.get("uw_task", True))
                extra_uw = [("consistency", loss_cons_total)] if (self.loss_weighting == "uw" and use_uw_task) else None

                out = self._loss_supervised(
                    stage=stage,
                    head_type=head_type,
                    pred_reg3=pred_reg3_c,
                    z=z_c,
                    z_layers=z_layers_c,
                    ratio_logits_pred=ratio_logits_pred_c,
                    pred_reg3_layers=pred_reg3_layers_c,
                    ratio_logits_layers=ratio_logits_layers_c,
                    ndvi_pred_from_head=ndvi_pred_c,
                    batch=batch_mixed,
                    extra_uw_losses=extra_uw,
                )
                if self.loss_weighting != "uw" or (not use_uw_task):
                    out["loss"] = out["loss"] + loss_cons_total

            else:
                # No consistency branches: fall back to standard supervised loss on the clean view.
                out = self._loss_supervised(
                    stage=stage,
                    head_type=head_type,
                    pred_reg3=pred_reg3_c,
                    z=z_c,
                    z_layers=z_layers_c,
                    ratio_logits_pred=ratio_logits_pred_c,
                    pred_reg3_layers=pred_reg3_layers_c,
                    ratio_logits_layers=ratio_logits_layers_c,
                    ndvi_pred_from_head=ndvi_pred_c,
                    batch=batch_mixed,
                )

            return out

        # --- Single-view (legacy) path ---
        images: Tensor = images_any  # type: ignore[assignment]
        if use_cutmix and self._cutmix_main is not None:
            try:
                batch, _ = self._cutmix_main.apply_main_batch(batch, force=True)  # type: ignore[assignment]
                images = batch["image"]
            except Exception:
                pass

        # Optional debug: dump the final input images actually fed to the model (after CutMix).
        self._maybe_dump_model_inputs(
            images=images,
            batch=batch,
            stage=stage,
            use_cutmix=use_cutmix,
            is_ndvi_only=is_ndvi_only,
        )

        head_type = str(getattr(self, "_head_type", "mlp")).lower()
        (
            pred_reg3,
            z,
            z_layers,
            ratio_logits_pred,
            ndvi_pred,
            batch,
            pred_reg3_layers,
            ratio_logits_layers,
        ) = self._predict_reg3_and_z(
            images=images,
            batch=batch,
            stage=stage,
            use_mixup=use_mixup,
            is_ndvi_only=is_ndvi_only,
        )

        if is_ndvi_only:
            return self._loss_ndvi_only(
                stage=stage,
                head_type=head_type,
                z=z,
                z_layers=z_layers,
                ndvi_pred_from_head=ndvi_pred,
                batch=batch,
            )

        return self._loss_supervised(
            stage=stage,
            head_type=head_type,
            pred_reg3=pred_reg3,
            z=z,
            z_layers=z_layers,
            ratio_logits_pred=ratio_logits_pred,
            pred_reg3_layers=pred_reg3_layers,
            ratio_logits_layers=ratio_logits_layers,
            ndvi_pred_from_head=ndvi_pred,
            batch=batch,
        )


