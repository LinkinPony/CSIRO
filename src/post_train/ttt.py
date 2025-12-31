from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
except Exception:  # pragma: no cover
    pl = None  # type: ignore[assignment]
    TensorBoardLogger = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PostTrainConfig:
    """
    Configuration for post-training (TTT / transductive adaptation).

    Notes:
      - This is intentionally minimal and driven by dict-based YAML config in the repo.
      - All values are interpreted as best-effort; missing/invalid values fall back to safe defaults.
    """

    enabled: bool = False
    force: bool = False

    # Optimization
    steps: int = 200
    batch_size: int = 4
    num_workers: int = 4
    lr_head: float = 1e-4
    lr_lora: float = 5e-5
    weight_decay_head: float = 0.0
    weight_decay_lora: float = 0.0
    max_grad_norm: float = 1.0

    # Loss weights
    weight_reg3: float = 1.0
    weight_ratio: float = 1.0

    # Teacher EMA (over trainable params only)
    ema_enabled: bool = True
    ema_decay: float = 0.999

    # Anchor regularization to pre-adaptation params (optional, protects against drift)
    anchor_weight: float = 0.0

    # Data / augmentation
    # Augment config follows the same schema as `data.augment` in train.yaml.
    augment_cfg: Dict[str, Any] = None  # type: ignore[assignment]

    # Logging
    log_every: int = 50
    # TensorBoard logging (best-effort). Logs will be written under:
    #   <out_dir>/tensorboard/<run_id>/
    # where <out_dir> is the directory containing the adapted head `.pt`.
    tensorboard: bool = True

    # Reproducibility
    seed: int = 42


def _now_compact() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_bool(v: Any, default: bool) -> bool:
    try:
        return bool(v)
    except Exception:
        return bool(default)


def _device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int) -> None:
    s = int(seed)
    torch.manual_seed(s)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass


class UnlabeledImageViewsDataset(Dataset):
    """
    Unlabeled image dataset that returns a dict with:
      - image: Tensor or tuple(Tensor, Tensor[, Tensor]) depending on transform
      - rel_path: relative path (for debugging / reproducibility)
    """

    def __init__(self, *, dataset_root: str, image_paths: Sequence[str], transform: Any) -> None:
        super().__init__()
        self.dataset_root = str(dataset_root)
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.image_paths))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from PIL import Image

        rel = str(self.image_paths[int(idx)])
        p = os.path.join(self.dataset_root, rel)
        img = Image.open(p).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        return {"image": x, "rel_path": rel}


def _mark_only_lora_trainable(m: nn.Module) -> int:
    """
    Best-effort: enable grads for LoRA parameters only; freeze everything else.
    Returns number of trainable params enabled.
    """
    n = 0
    for name, p in m.named_parameters():
        is_lora = ("lora_" in name) or ("lora_magnitude_vector" in name)
        p.requires_grad = bool(is_lora)
        if p.requires_grad:
            n += 1
    return n


class _PostTrainLightningModule(nn.Module):
    """
    Small wrapper to present a unified forward for (feature_extractor, head).
    Returns:
      - reg3 logits: Tensor[B, 3]
      - ratio logits: Tensor[B, 3] or None
    """

    def __init__(self, *, feature_extractor: nn.Module, head: nn.Module, meta: Dict[str, Any]) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.meta = dict(meta)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return _forward_preds(
            feature_extractor=self.feature_extractor,
            head=self.head,
            meta=self.meta,
            images=images,
        )


if pl is not None:

    class _TTTLightningModule(pl.LightningModule):
        """
        Lightning implementation of post-train (TTT) that keeps the same objective as the
        previous manual loop, but delegates:
          - optimizer step / grad scaling (AMP)
          - gradient clipping
          - logging to TensorBoard
        to Lightning.
        """

        def __init__(
            self,
            *,
            student: _PostTrainLightningModule,
            tracked: List[Tuple[str, nn.Parameter]],
            cfg: PostTrainConfig,
        ) -> None:
            super().__init__()
            self.student = student
            self.cfg_pt = cfg
            self._tracked = tracked

            # EMA buffers over trainable params only (keeps overhead small).
            self._ema: Dict[str, torch.Tensor] = {}
            self._init: Dict[str, torch.Tensor] = {}
            # Cached list of params for optimizer grouping (resolved once).
            self._head_params: List[nn.Parameter] = [p for p in self.student.head.parameters() if p.requires_grad]
            self._lora_params: List[nn.Parameter] = [
                p
                for n, p in self.student.feature_extractor.backbone.named_parameters()  # type: ignore[attr-defined]
                if p.requires_grad and (("lora_" in n) or ("lora_magnitude_vector" in n))
            ]

        def on_fit_start(self) -> None:
            # Initialize EMA and anchor snapshots on the correct device.
            with torch.no_grad():
                self._ema = {}
                self._init = {}
                for name, p in self._tracked:
                    self._ema[name] = p.detach().clone().to(dtype=torch.float32, device=p.device)
                    if float(self.cfg_pt.anchor_weight) > 0.0:
                        self._init[name] = p.detach().clone().to(dtype=torch.float32, device=p.device)

        def configure_optimizers(self):
            param_groups: List[Dict[str, Any]] = []
            if self._head_params:
                param_groups.append(
                    {
                        "params": self._head_params,
                        "lr": float(self.cfg_pt.lr_head),
                        "weight_decay": float(self.cfg_pt.weight_decay_head),
                    }
                )
            if self._lora_params:
                param_groups.append(
                    {
                        "params": self._lora_params,
                        "lr": float(self.cfg_pt.lr_lora),
                        "weight_decay": float(self.cfg_pt.weight_decay_lora),
                    }
                )
            if not param_groups:
                raise RuntimeError("No trainable parameters found for post-train (head and LoRA are both frozen).")
            return torch.optim.AdamW(param_groups)

        def _swap_teacher_params_(self, stash: Dict[str, torch.Tensor]) -> None:
            # Swap EMA weights into live params (trainable only).
            for name, p in self._tracked:
                stash[name] = p.detach().clone()
                p.data.copy_(self._ema[name].to(device=p.device, dtype=p.dtype))  # type: ignore[call-arg]

        def _restore_teacher_params_(self, stash: Dict[str, torch.Tensor]) -> None:
            for name, p in self._tracked:
                prev = stash.get(name, None)
                if prev is None:
                    continue
                p.data.copy_(prev.to(device=p.device, dtype=p.dtype))  # type: ignore[call-arg]

        def _update_ema_(self) -> None:
            if not bool(self.cfg_pt.ema_enabled):
                return
            d = float(self.cfg_pt.ema_decay)
            if not (0.0 < d < 1.0):
                return
            with torch.no_grad():
                for name, p in self._tracked:
                    v = p.detach().to(dtype=torch.float32, device=self._ema[name].device)
                    self._ema[name].mul_(d).add_(v, alpha=(1.0 - d))

        def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
            images_any = batch.get("image")
            if isinstance(images_any, (list, tuple)) and len(images_any) >= 2:
                images_clean = images_any[0]
                images_aug_list = list(images_any[1:])
            else:
                images_clean = images_any
                images_aug_list = [images_any]

            if not isinstance(images_clean, torch.Tensor):
                raise RuntimeError("Post-train dataset must return Tensor images (or tuple/list of Tensors).")

            # --- Teacher prediction (EMA weights, eval, no grad) ---
            stash: Dict[str, torch.Tensor] = {}
            backbone_was_train = bool(getattr(self.student.feature_extractor.backbone, "training", False))  # type: ignore[attr-defined]
            head_was_train = bool(getattr(self.student.head, "training", False))
            if bool(self.cfg_pt.ema_enabled):
                self._swap_teacher_params_(stash)
            self.student.feature_extractor.backbone.eval()  # type: ignore[attr-defined]
            self.student.head.eval()
            with torch.no_grad():
                t_reg3, t_ratio = self.student(images_clean)
                t_reg3 = t_reg3.detach()
                t_ratio = t_ratio.detach() if isinstance(t_ratio, torch.Tensor) else None
            # Restore train mode + weights
            self.student.feature_extractor.backbone.train(backbone_was_train)  # type: ignore[attr-defined]
            self.student.head.train(head_was_train)
            if bool(self.cfg_pt.ema_enabled):
                self._restore_teacher_params_(stash)

            # --- Student prediction(s) ---
            loss_reg3 = torch.zeros((), device=self.device)
            loss_ratio = torch.zeros((), device=self.device)
            k = 0
            for img_aug in images_aug_list:
                if not isinstance(img_aug, torch.Tensor):
                    continue
                s_reg3, s_ratio = self.student(img_aug)
                loss_reg3 = loss_reg3 + F.mse_loss(s_reg3, t_reg3)
                if t_ratio is not None and isinstance(s_ratio, torch.Tensor):
                    loss_ratio = loss_ratio + F.mse_loss(s_ratio, t_ratio)
                k += 1
            if k > 1:
                loss_reg3 = loss_reg3 / float(k)
                loss_ratio = loss_ratio / float(k)

            loss = (float(self.cfg_pt.weight_reg3) * loss_reg3) + (float(self.cfg_pt.weight_ratio) * loss_ratio)

            # Optional anchor
            if float(self.cfg_pt.anchor_weight) > 0.0 and self._init:
                lam = float(self.cfg_pt.anchor_weight)
                reg = torch.zeros((), device=self.device)
                for name, p in self._tracked:
                    p0 = self._init.get(name, None)
                    if p0 is None:
                        continue
                    diff = p.float() - p0.to(device=p.device, dtype=torch.float32)
                    reg = reg + diff.pow(2).mean()
                loss = loss + (lam * reg)

            # Lightning logging (goes to TensorBoardLogger if enabled)
            self.log("loss/total", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log("loss/reg3", loss_reg3, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log("loss/ratio", loss_ratio, on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log("lr/head", float(self.cfg_pt.lr_head), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            self.log("lr/lora", float(self.cfg_pt.lr_lora), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            return loss

        def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
            # Update EMA after optimizer step.
            self._update_ema_()


def _collect_trainable_named_params(
    *,
    head: nn.Module,
    backbone: nn.Module,
) -> List[Tuple[str, nn.Parameter]]:
    """
    Return a stable list of (name, param) for trainable params in the post-train graph.

    - head: all parameters
    - backbone: LoRA-only parameters (requires_grad True expected)
    """
    named: List[Tuple[str, nn.Parameter]] = []
    for n, p in head.named_parameters():
        if p.requires_grad:
            named.append((f"head.{n}", p))
    for n, p in backbone.named_parameters():
        if p.requires_grad:
            named.append((f"backbone.{n}", p))
    # Stable order for reproducibility
    named.sort(key=lambda kv: kv[0])
    return named


def _swap_params_(
    params: List[Tuple[str, nn.Parameter]],
    *,
    new_values: Dict[str, torch.Tensor],
    stash: Dict[str, torch.Tensor],
) -> None:
    """
    In-place swap param.data with provided tensors, stashing originals into `stash`.
    """
    for name, p in params:
        if name not in new_values:
            continue
        stash[name] = p.data
        p.data = new_values[name].to(device=p.data.device, dtype=p.data.dtype)


def _restore_params_(
    params: List[Tuple[str, nn.Parameter]],
    *,
    stash: Dict[str, torch.Tensor],
) -> None:
    for name, p in params:
        if name in stash:
            p.data = stash[name]


def _build_head_module_from_meta(meta: Dict[str, Any], cfg_train_yaml: Dict[str, Any]) -> nn.Module:
    """
    Build a head module matching the serialized head `state_dict`.

    This mirrors `src/inference/pipeline.py` head builder logic (subset).
    """
    from src.models.head_builder import MultiLayerHeadExport, build_head_layer
    from src.models.spatial_fpn import FPNHeadConfig, FPNScalarHead
    from src.models.vitdet_head import ViTDetHeadConfig, ViTDetMultiLayerScalarHead, ViTDetScalarHead
    from src.models.dpt_scalar_head import DPTHeadConfig, DPTScalarHead

    meta = dict(meta or {})
    head_type = str(meta.get("head_type", "mlp") or "mlp").strip().lower()

    # Common meta
    num_main = int(meta.get("num_outputs_main", meta.get("num_outputs", 1)))
    num_ratio = int(meta.get("num_outputs_ratio", 0))
    head_total = int(meta.get("head_total_outputs", num_main + num_ratio))
    # Some exported heads may be "main-only" even if num_outputs_ratio is present in meta.
    # Keep parity with the inference builder by enabling ratio outputs only when the packed
    # dimension matches main+ratio.
    head_is_ratio = bool(num_ratio > 0 and head_total == (num_main + num_ratio))
    num_ratio_eff = int(num_ratio) if head_is_ratio else 0
    embedding_dim = int(meta.get("embedding_dim", int(cfg_train_yaml.get("model", {}).get("embedding_dim", 1024))))
    head_hidden_dims = list(
        meta.get(
            "head_hidden_dims",
            cfg_train_yaml.get("model", {}).get("head", {}).get("hidden_dims", [512, 256]),
        )
    )
    head_activation = str(
        meta.get(
            "head_activation",
            cfg_train_yaml.get("model", {}).get("head", {}).get("activation", "relu"),
        )
    )
    head_dropout = float(
        meta.get(
            "head_dropout",
            cfg_train_yaml.get("model", {}).get("head", {}).get("dropout", 0.0),
        )
    )

    use_layerwise_heads = bool(meta.get("use_layerwise_heads", False))
    backbone_layer_indices = list(meta.get("backbone_layer_indices", []))
    use_separate_bottlenecks = bool(meta.get("use_separate_bottlenecks", False))
    num_layers_eff = max(1, len(backbone_layer_indices)) if use_layerwise_heads else 1

    # Ratio coupling flags (kept for head builders)
    separate_ratio_head = bool(meta.get("separate_ratio_head", False))
    separate_ratio_spatial_head = bool(meta.get("separate_ratio_spatial_head", False))

    if head_type == "fpn":
        fpn_dim = int(meta.get("fpn_dim", int(cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_dim", 256))))
        fpn_levels = int(meta.get("fpn_num_levels", int(cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_num_levels", 3))))
        fpn_patch_size = int(meta.get("fpn_patch_size", int(cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_patch_size", 16))))
        fpn_reverse = bool(meta.get("fpn_reverse_level_order", bool(cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_reverse_level_order", True))))
        enable_ndvi = bool(meta.get("enable_ndvi", False))
        return FPNScalarHead(
            FPNHeadConfig(
                embedding_dim=embedding_dim,
                fpn_dim=fpn_dim,
                num_levels=fpn_levels,
                num_layers=num_layers_eff,
                use_separate_bottlenecks=use_separate_bottlenecks,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
                num_outputs_main=num_main,
                num_outputs_ratio=num_ratio_eff,
                enable_ndvi=enable_ndvi,
                separate_ratio_head=separate_ratio_head,
                separate_ratio_spatial_head=separate_ratio_spatial_head,
                patch_size=fpn_patch_size,
                reverse_level_order=fpn_reverse,
            )
        )

    if head_type == "vitdet":
        vitdet_dim = int(meta.get("vitdet_dim", int(cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_dim", 256))))
        vitdet_patch_size = int(
            meta.get(
                "vitdet_patch_size",
                int(cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_patch_size", cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_patch_size", 16))),
            )
        )
        vitdet_scale_factors = list(meta.get("vitdet_scale_factors", cfg_train_yaml.get("model", {}).get("head", {}).get("vitdet_scale_factors", [2.0, 1.0, 0.5])))
        enable_ndvi = bool(meta.get("enable_ndvi", False))
        vitdet_cfg = ViTDetHeadConfig(
            embedding_dim=embedding_dim,
            vitdet_dim=vitdet_dim,
            scale_factors=vitdet_scale_factors,
            patch_size=vitdet_patch_size,
            num_outputs_main=num_main,
            num_outputs_ratio=num_ratio_eff,
            enable_ndvi=enable_ndvi,
            separate_ratio_head=separate_ratio_head,
            separate_ratio_spatial_head=separate_ratio_spatial_head,
            head_hidden_dims=head_hidden_dims,
            head_activation=head_activation,
            dropout=head_dropout,
        )
        if use_layerwise_heads:
            fusion_mode = str(meta.get("backbone_layers_fusion", meta.get("layer_fusion", "mean")) or "mean").strip().lower()
            return ViTDetMultiLayerScalarHead(vitdet_cfg, num_layers=num_layers_eff, layer_fusion=fusion_mode)
        return ViTDetScalarHead(vitdet_cfg)

    if head_type == "dpt":
        dpt_features = int(meta.get("dpt_features", int(cfg_train_yaml.get("model", {}).get("head", {}).get("dpt_features", 256))))
        dpt_patch_size = int(meta.get("dpt_patch_size", int(cfg_train_yaml.get("model", {}).get("head", {}).get("dpt_patch_size", cfg_train_yaml.get("model", {}).get("head", {}).get("fpn_patch_size", 16)))))
        dpt_readout = str(meta.get("dpt_readout", cfg_train_yaml.get("model", {}).get("head", {}).get("dpt_readout", "ignore"))).strip().lower()
        enable_ndvi = bool(meta.get("enable_ndvi", False))
        return DPTScalarHead(
            DPTHeadConfig(
                embedding_dim=embedding_dim,
                features=dpt_features,
                patch_size=dpt_patch_size,
                readout=dpt_readout,
                num_layers=num_layers_eff,
                num_outputs_main=num_main,
                num_outputs_ratio=num_ratio_eff,
                enable_ndvi=enable_ndvi,
                separate_ratio_head=separate_ratio_head,
                separate_ratio_spatial_head=separate_ratio_spatial_head,
                head_hidden_dims=head_hidden_dims,
                head_activation=head_activation,
                dropout=head_dropout,
            )
        )

    # MLP-style heads (packed)
    use_patch_reg3 = bool(meta.get("use_patch_reg3", False))
    use_cls_token = bool(meta.get("use_cls_token", True))

    if use_layerwise_heads and use_separate_bottlenecks:
        return MultiLayerHeadExport(
            embedding_dim=embedding_dim,
            num_outputs_main=num_main,
            num_outputs_ratio=num_ratio_eff,
            head_hidden_dims=head_hidden_dims,
            head_activation=head_activation,
            dropout=head_dropout,
            use_patch_reg3=use_patch_reg3,
            use_cls_token=use_cls_token,
            num_layers=num_layers_eff,
        )

    # Packed single module (may encode per-layer outputs in the final linear layer).
    effective_outputs = head_total if not use_layerwise_heads else head_total * num_layers_eff
    # Patch-mode uses embedding_dim input. Global mode uses 2C when CLS is included, otherwise C.
    input_dim = embedding_dim if (use_patch_reg3 or (not use_cls_token)) else None
    return build_head_layer(
        embedding_dim=embedding_dim,
        num_outputs=effective_outputs,
        head_hidden_dims=head_hidden_dims,
        head_activation=head_activation,
        dropout=head_dropout,
        use_output_softplus=False,
        input_dim=input_dim,
    )


def _forward_preds(
    *,
    feature_extractor: nn.Module,
    head: nn.Module,
    meta: Dict[str, Any],
    images: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward path that returns:
      - reg3_pred: (B, num_outputs_main)
      - ratio_logits: (B, num_outputs_ratio) or None
    """
    head_type = str(meta.get("head_type", "mlp") or "mlp").strip().lower()
    use_layerwise_heads = bool(meta.get("use_layerwise_heads", False))
    layer_indices = list(meta.get("backbone_layer_indices", []))
    use_separate_bottlenecks = bool(meta.get("use_separate_bottlenecks", False))
    fusion_mode = str(meta.get("backbone_layers_fusion", meta.get("layer_fusion", "mean")) or "mean").strip().lower()
    layer_weights: Optional[torch.Tensor] = None
    if use_layerwise_heads and fusion_mode == "learned":
        try:
            logits_meta = meta.get("mlp_layer_logits", None)
            if isinstance(logits_meta, (list, tuple)) and len(logits_meta) == len(layer_indices):
                logits_t = torch.tensor([float(x) for x in logits_meta], device=images.device, dtype=torch.float32)
                layer_weights = torch.softmax(logits_t, dim=0).to(dtype=images.dtype)
        except Exception:
            layer_weights = None

    num_main = int(meta.get("num_outputs_main", meta.get("num_outputs", 1)))
    num_ratio_raw = int(meta.get("num_outputs_ratio", 0))
    head_total = int(meta.get("head_total_outputs", num_main + num_ratio_raw))
    head_is_ratio = bool(num_ratio_raw > 0 and head_total == (num_main + num_ratio_raw))
    num_ratio = int(num_ratio_raw) if head_is_ratio else 0

    H = int(images.shape[-2])
    W = int(images.shape[-1])

    if head_type in ("fpn", "vitdet"):
        if use_layerwise_heads and layer_indices:
            _cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
            pt_in: Any = pt_list
        else:
            _cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
            pt_in = pt
        out = head(pt_in, image_hw=(H, W))  # type: ignore[call-arg]
        if not isinstance(out, dict) or "reg3" not in out:
            raise RuntimeError("Head forward did not return dict with key 'reg3'")
        reg3 = out["reg3"]
        ratio = out.get("ratio", None)
        return reg3, ratio

    if head_type == "dpt":
        if use_layerwise_heads and layer_indices:
            cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
            out = head(cls_list, pt_list, image_hw=(H, W))  # type: ignore[call-arg]
        else:
            cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
            out = head(cls, pt, image_hw=(H, W))  # type: ignore[call-arg]
        if not isinstance(out, dict) or "reg3" not in out:
            raise RuntimeError("DPT head forward did not return dict with key 'reg3'")
        reg3 = out["reg3"]
        ratio = out.get("ratio", None)
        return reg3, ratio

    # --- MLP packed head(s) ---
    use_patch_reg3 = bool(meta.get("use_patch_reg3", False))
    use_cls_token = bool(meta.get("use_cls_token", True))

    # Multi-layer MLP path
    if use_layerwise_heads and layer_indices:
        cls_list, pt_list = feature_extractor.forward_layers_cls_and_tokens(images, layer_indices)  # type: ignore[attr-defined]
        num_layers = len(pt_list)
        if num_layers <= 0:
            raise RuntimeError("Empty pt_list in multi-layer MLP path")

        main_layers: List[torch.Tensor] = []
        ratio_layers: List[torch.Tensor] = []

        for l_idx, (cls_l, pt_l) in enumerate(zip(cls_list, pt_list)):
            if not use_patch_reg3:
                patch_mean = pt_l.mean(dim=1)
                if use_separate_bottlenecks and hasattr(head, "forward_global_layer"):
                    layer_main, layer_ratio = head.forward_global_layer(cls_l, patch_mean, l_idx)  # type: ignore[attr-defined]
                else:
                    feats = torch.cat([cls_l, patch_mean], dim=-1) if use_cls_token else patch_mean
                    out_all = head(feats)
                    # out_all is packed (B, head_total * num_layers)
                    if out_all.shape[-1] != head_total * num_layers:
                        raise RuntimeError("Unexpected packed head dimension in multi-layer MLP forward")
                    sl = out_all[:, l_idx * head_total : (l_idx + 1) * head_total]
                    layer_main = sl[:, :num_main]
                    layer_ratio = sl[:, num_main : num_main + num_ratio] if num_ratio > 0 else None
            else:
                # Patch mode:
                # - main: apply head on each patch token and average over patches
                # - ratio: apply head on mean patch token
                if use_separate_bottlenecks and hasattr(head, "forward_patch_layer"):
                    layer_main, layer_ratio = head.forward_patch_layer(pt_l, l_idx)  # type: ignore[attr-defined]
                else:
                    B, N, C = pt_l.shape
                    flat = pt_l.reshape(B * N, C)
                    out_all_patch = head(flat)
                    if out_all_patch.shape[-1] != head_total * num_layers:
                        raise RuntimeError("Unexpected packed head dimension in multi-layer patch-mode forward")
                    slp = out_all_patch[:, l_idx * head_total : (l_idx + 1) * head_total]
                    layer_main = slp[:, :num_main].view(B, N, num_main).mean(dim=1)
                    if num_ratio > 0:
                        patch_mean = pt_l.mean(dim=1)
                        out_all_g = head(patch_mean)
                        slg = out_all_g[:, l_idx * head_total : (l_idx + 1) * head_total]
                        layer_ratio = slg[:, num_main : num_main + num_ratio]
                    else:
                        layer_ratio = None

            main_layers.append(layer_main)
            if layer_ratio is not None:
                ratio_layers.append(layer_ratio)

        # Fuse over layers
        main_stack = torch.stack(main_layers, dim=0)  # (L,B,D)
        if layer_weights is None:
            reg3 = main_stack.mean(dim=0)
        else:
            w = layer_weights.view(-1, 1, 1)
            reg3 = (main_stack * w).sum(dim=0)

        ratio: Optional[torch.Tensor]
        if num_ratio > 0 and ratio_layers:
            ratio_stack = torch.stack(ratio_layers, dim=0)  # (K,B,R)
            if layer_weights is None or ratio_stack.shape[0] != layer_weights.numel():
                ratio = ratio_stack.mean(dim=0)
            else:
                w = layer_weights.view(-1, 1, 1)
                ratio = (ratio_stack * w).sum(dim=0)
        else:
            ratio = None
        return reg3, ratio

    # Single-layer MLP path
    if not use_patch_reg3:
        if use_cls_token:
            feats = feature_extractor(images)  # type: ignore[operator]
        else:
            _cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
            feats = pt.mean(dim=1)
        out = head(feats)
        reg3 = out[:, :num_main]
        ratio = out[:, num_main : num_main + num_ratio] if num_ratio > 0 else None
        return reg3, ratio

    # Patch-mode single-layer
    _cls, pt = feature_extractor.forward_cls_and_tokens(images)  # type: ignore[attr-defined]
    B, N, C = pt.shape
    flat = pt.reshape(B * N, C)
    out_all_patch = head(flat)  # (B*N, head_total)
    reg3 = out_all_patch[:, :num_main].view(B, N, num_main).mean(dim=1)
    ratio: Optional[torch.Tensor]
    if num_ratio > 0:
        patch_mean = pt.mean(dim=1)
        out_g = head(patch_mean)
        ratio = out_g[:, num_main : num_main + num_ratio]
    else:
        ratio = None
    return reg3, ratio


def post_train_single_head(
    *,
    cfg_train_yaml: Dict[str, Any],
    dino_weights_pt_file: str,
    head_in_pt: str,
    dataset_root: str,
    image_paths: Sequence[str],
    out_head_pt: str,
    cfg: PostTrainConfig,
) -> str:
    """
    Post-train (adapt) a single head package on unlabeled images.

    Returns:
        Path to the written adapted head package (.pt).
    """
    out_path = str(out_head_pt)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    if (not cfg.force) and os.path.isfile(out_path):
        return out_path

    _set_seed(cfg.seed)
    device = _device_auto()

    from src.inference.torch_load import load_head_state
    from src.models.backbone import build_feature_extractor
    from src.models.peft_integration import export_lora_payload_if_any

    head_state, head_meta, peft_payload = load_head_state(str(head_in_pt))
    head_meta = dict(head_meta or {})

    # Build feature extractor (DINO backbone wrapped)
    backbone_name = str(cfg_train_yaml.get("model", {}).get("backbone", "") or "").strip()
    if not backbone_name:
        raise RuntimeError("cfg_train_yaml missing model.backbone (cannot build backbone).")
    feature_extractor = build_feature_extractor(
        backbone_name=backbone_name,
        pretrained=bool(cfg_train_yaml.get("model", {}).get("pretrained", True)),
        weights_url=str(cfg_train_yaml.get("model", {}).get("weights_url", "") or "") or None,
        weights_path=str(dino_weights_pt_file),
        gradient_checkpointing=bool(cfg_train_yaml.get("model", {}).get("gradient_checkpointing", False)),
    )
    # Enable grads through the backbone forward for LoRA training.
    try:
        feature_extractor.inference_only = False  # type: ignore[attr-defined]
    except Exception:
        pass

    # Inject LoRA from payload (preferred) or via config (fallback)
    lora_params_enabled = 0
    if peft_payload is not None and isinstance(peft_payload, dict):
        peft_cfg_dict = peft_payload.get("config", None)
        peft_state = peft_payload.get("state_dict", None)
        if peft_cfg_dict and peft_state:
            try:
                from peft.config import PeftConfig  # type: ignore
                from peft.mapping_func import get_peft_model  # type: ignore
                from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore
            except Exception:
                from src.models.peft_integration import _import_peft

                _import_peft()
                from peft.config import PeftConfig  # type: ignore
                from peft.mapping_func import get_peft_model  # type: ignore
                from peft.utils.save_and_load import set_peft_model_state_dict  # type: ignore

            peft_config = PeftConfig.from_peft_type(**peft_cfg_dict)
            feature_extractor.backbone = get_peft_model(feature_extractor.backbone, peft_config)  # type: ignore[assignment]
            set_peft_model_state_dict(feature_extractor.backbone, peft_state, adapter_name="default")  # type: ignore[arg-type]
            lora_params_enabled = _mark_only_lora_trainable(feature_extractor.backbone)
    else:
        # Fallback: allow user to post-train with LoRA even when no payload exists.
        peft_cfg = dict(cfg_train_yaml.get("peft", {}) or {})
        if bool(peft_cfg.get("enabled", False)):
            from src.models.peft_integration import inject_lora_into_feature_extractor

            feature_extractor, _ = inject_lora_into_feature_extractor(feature_extractor, peft_cfg)  # type: ignore[assignment]
            lora_params_enabled = _mark_only_lora_trainable(feature_extractor.backbone)

    # Build head module and load weights
    head_module = _build_head_module_from_meta(head_meta, cfg_train_yaml)
    head_module.load_state_dict(head_state, strict=True)

    # Move modules to device
    feature_extractor = feature_extractor.to(device)
    head_module = head_module.to(device)

    # Put modules in train mode for student path
    try:
        feature_extractor.backbone.train()  # type: ignore[attr-defined]
    except Exception:
        feature_extractor.train()
    head_module.train()

    # Build transform for unlabeled multi-view training
    from src.data.augmentations import build_train_transform

    def _parse_image_size(value: Any) -> Tuple[int, int]:
        """
        Accept int (square) or [width, height]; return (height, width).
        Keep consistent with `src.training.single_run.parse_image_size`.
        """
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                w, h = int(value[0]), int(value[1])
                return (int(h), int(w))
            v = int(value)
            return (v, v)
        except Exception:
            v = int(value)
            return (v, v)

    image_size = _parse_image_size(cfg_train_yaml.get("data", {}).get("image_size", 224))
    mean = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("mean", [0.485, 0.456, 0.406]))
    std = list(cfg_train_yaml.get("data", {}).get("normalization", {}).get("std", [0.229, 0.224, 0.225]))
    aug_cfg = dict(cfg.augment_cfg or {})
    tf = build_train_transform(image_size=image_size, mean=mean, std=std, augment_cfg=aug_cfg)

    ds = UnlabeledImageViewsDataset(
        dataset_root=str(dataset_root),
        image_paths=list(image_paths),
        transform=tf,
    )
    dl = DataLoader(
        ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=torch.cuda.is_available(),
        drop_last=bool(len(ds) >= max(2, int(cfg.batch_size))),
        persistent_workers=bool(int(cfg.num_workers) > 0),
    )

    # Optimizer (separate LR for head vs LoRA)
    head_params = [p for p in head_module.parameters() if p.requires_grad]
    lora_params = [p for n, p in feature_extractor.backbone.named_parameters() if p.requires_grad and (("lora_" in n) or ("lora_magnitude_vector" in n))]  # type: ignore[attr-defined]
    if not lora_params:
        # Still allow head-only adaptation; but user requested LoRA+head so we warn loudly.
        print(f"[POST_TRAIN][WARN] No trainable LoRA params found for head={head_in_pt}. LoRA may be disabled.")
    else:
        # Keep counter for diagnostics (can differ from lora_params_enabled if PEFT wrapper changes naming)
        _ = lora_params_enabled

    param_groups: List[Dict[str, Any]] = []
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": float(cfg.lr_head),
                "weight_decay": float(cfg.weight_decay_head),
            }
        )
    if lora_params:
        param_groups.append(
            {
                "params": lora_params,
                "lr": float(cfg.lr_lora),
                "weight_decay": float(cfg.weight_decay_lora),
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for post-train (head and LoRA are both frozen).")

    optimizer = torch.optim.AdamW(param_groups)

    # EMA state over trainable params
    tracked = _collect_trainable_named_params(head=head_module, backbone=feature_extractor.backbone)  # type: ignore[arg-type]
    ema: Dict[str, torch.Tensor] = {}
    init: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, p in tracked:
            # Keep EMA on the same device as the parameter to avoid CPU<->GPU traffic each step.
            ema[name] = p.detach().clone().to(dtype=torch.float32, device=p.device)
            # Only keep an anchor snapshot when enabled (saves memory).
            if float(cfg.anchor_weight) > 0.0:
                init[name] = p.detach().clone().to(dtype=torch.float32, device=p.device)

    def _update_ema() -> None:
        if not bool(cfg.ema_enabled):
            return
        d = float(cfg.ema_decay)
        if not (0.0 < d < 1.0):
            return
        with torch.no_grad():
            for name, p in tracked:
                v = p.detach().to(dtype=torch.float32, device=ema[name].device)
                ema[name].mul_(d).add_(v, alpha=(1.0 - d))

    # Simple iterator cycling
    it: Iterator[Dict[str, Any]] = iter(dl)

    def _next_batch() -> Dict[str, Any]:
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)

    # Training loop (Lightning)
    n_steps = max(0, int(cfg.steps))
    if n_steps <= 0:
        # Still export a copy (useful for consistent packaging).
        state_out: Dict[str, Any] = {
            "state_dict": head_module.state_dict(),
            "meta": dict(head_meta),
        }
        try:
            peft_out = export_lora_payload_if_any(feature_extractor.backbone)  # type: ignore[arg-type]
            if peft_out is not None:
                state_out["peft"] = peft_out
        except Exception:
            pass
        torch.save(state_out, out_path)
        return out_path

    if pl is None:
        raise RuntimeError("lightning.pytorch is required for post_train but could not be imported.")

    run_id = f"post_train_{_now_compact()}"
    tb_logger = None
    if bool(getattr(cfg, "tensorboard", True)) and TensorBoardLogger is not None:
        try:
            # Write to: <out_dir>/tensorboard/<run_id>/
            tb_logger = TensorBoardLogger(save_dir=str(out_dir), name="tensorboard", version=run_id)
            try:
                tb_logger.log_hyperparams(
                    {
                        "base_head": str(head_in_pt),
                        "out_head": str(out_path),
                        "steps": int(cfg.steps),
                        "batch_size": int(cfg.batch_size),
                        "lr_head": float(cfg.lr_head),
                        "lr_lora": float(cfg.lr_lora),
                        "ema_enabled": bool(cfg.ema_enabled),
                        "ema_decay": float(cfg.ema_decay),
                        "anchor_weight": float(cfg.anchor_weight),
                    }
                )
            except Exception:
                pass
        except Exception:
            tb_logger = None

    # Build Lightning module around the existing modules
    student = _PostTrainLightningModule(feature_extractor=feature_extractor, head=head_module, meta=head_meta)
    ttt_module = _TTTLightningModule(student=student, tracked=tracked, cfg=cfg)  # type: ignore[arg-type]

    # Choose precision for AMP
    use_cuda = bool(torch.cuda.is_available() and device.type == "cuda")
    precision = "16-mixed" if use_cuda else 32

    # Ensure enough epochs to reach max_steps, without needing an infinite dataloader.
    try:
        n_batches = max(1, int(len(dl)))
    except Exception:
        n_batches = 1
    max_epochs = max(1, int((n_steps + n_batches - 1) // n_batches))

    trainer = pl.Trainer(
        max_steps=int(n_steps),
        max_epochs=int(max_epochs),
        accelerator="auto",
        devices=1,
        precision=precision,
        logger=tb_logger if tb_logger is not None else False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=max(1, int(cfg.log_every)),
        num_sanity_val_steps=0,
        gradient_clip_val=float(cfg.max_grad_norm) if float(cfg.max_grad_norm) > 0.0 else 0.0,
        gradient_clip_algorithm="norm",
    )

    # Fit (train-only)
    trainer.fit(model=ttt_module, train_dataloaders=dl)

    # Export updated head package (plus LoRA payload)
    out_state: Dict[str, Any] = {
        "state_dict": head_module.state_dict(),
        "meta": dict(head_meta),
    }
    # Attach post-train metadata (non-breaking for inference)
    try:
        out_state["meta"]["post_train"] = {
            "base_head": str(head_in_pt),
            "steps": int(cfg.steps),
            "batch_size": int(cfg.batch_size),
            "lr_head": float(cfg.lr_head),
            "lr_lora": float(cfg.lr_lora),
            "weight_reg3": float(cfg.weight_reg3),
            "weight_ratio": float(cfg.weight_ratio),
            "ema_enabled": bool(cfg.ema_enabled),
            "ema_decay": float(cfg.ema_decay),
            "anchor_weight": float(cfg.anchor_weight),
            "timestamp": _now_compact(),
        }
    except Exception:
        pass

    try:
        peft_out = export_lora_payload_if_any(feature_extractor.backbone)  # type: ignore[arg-type]
        if peft_out is not None:
            out_state["peft"] = peft_out
    except Exception:
        pass

    # Atomic-ish save: write temp then rename.
    tmp_path = os.path.join(out_dir, f".tmp_post_train_{os.getpid()}_{int(time.time())}.pt")
    torch.save(out_state, tmp_path)
    os.replace(tmp_path, out_path)

    # Also emit a history snapshot (step-tagged) alongside infer_head.pt to support versioning.
    try:
        hist_name = f"head-post-step{int(cfg.steps):06d}-{_now_compact()}.pt"
        hist_path = os.path.join(out_dir, hist_name)
        # Best-effort copy (do not overwrite)
        if not os.path.isfile(hist_path):
            torch.save(out_state, hist_path)
    except Exception:
        pass

    return out_path


def parse_post_train_config(cfg_obj: Optional[Dict[str, Any]]) -> PostTrainConfig:
    """
    Parse a YAML-style dict (cfg.get('post_train')) into PostTrainConfig with safe defaults.
    """
    cfg_obj = dict(cfg_obj or {})
    ema = dict(cfg_obj.get("ema", {}) or {})
    loss = dict(cfg_obj.get("loss", {}) or {})
    anchor = dict(cfg_obj.get("anchor", {}) or {})
    aug = dict(cfg_obj.get("augment", cfg_obj.get("augment_cfg", {})) or {})

    return PostTrainConfig(
        enabled=_safe_bool(cfg_obj.get("enabled", False), False),
        force=_safe_bool(cfg_obj.get("force", False), False),
        steps=max(0, _safe_int(cfg_obj.get("steps", 200), 200)),
        batch_size=max(1, _safe_int(cfg_obj.get("batch_size", 4), 4)),
        num_workers=max(0, _safe_int(cfg_obj.get("num_workers", 4), 4)),
        lr_head=_safe_float(cfg_obj.get("lr_head", cfg_obj.get("lr", 1e-4)), 1e-4),
        lr_lora=_safe_float(cfg_obj.get("lr_lora", cfg_obj.get("lora_lr", 5e-5)), 5e-5),
        weight_decay_head=_safe_float(cfg_obj.get("weight_decay_head", cfg_obj.get("weight_decay", 0.0)), 0.0),
        weight_decay_lora=_safe_float(cfg_obj.get("weight_decay_lora", 0.0), 0.0),
        max_grad_norm=_safe_float(cfg_obj.get("max_grad_norm", 1.0), 1.0),
        weight_reg3=_safe_float(loss.get("weight_reg3", cfg_obj.get("weight_reg3", 1.0)), 1.0),
        weight_ratio=_safe_float(loss.get("weight_ratio", cfg_obj.get("weight_ratio", 1.0)), 1.0),
        ema_enabled=_safe_bool(ema.get("enabled", cfg_obj.get("ema_enabled", True)), True),
        ema_decay=_safe_float(ema.get("decay", cfg_obj.get("ema_decay", 0.999)), 0.999),
        anchor_weight=_safe_float(anchor.get("weight", cfg_obj.get("anchor_weight", 0.0)), 0.0),
        augment_cfg=dict(aug),
        log_every=max(0, _safe_int(cfg_obj.get("log_every", 50), 50)),
        seed=_safe_int(cfg_obj.get("seed", 42), 42),
    )


def ensure_parent_dir(path: str) -> None:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)



