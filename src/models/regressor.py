from typing import Any, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from lightning.pytorch import LightningModule
from loguru import logger

from .backbone import build_feature_extractor
from .peft_integration import inject_lora_into_feature_extractor, get_lora_param_list
from .head_builder import build_head_layer


class BiomassRegressor(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_outputs: int = 3,
        dropout: float = 0.0,
        head_hidden_dims: Optional[List[int]] = None,
        head_activation: str = "relu",
        use_output_softplus: bool = True,
        pretrained: bool = True,
        weights_url: Optional[str] = None,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: Optional[str] = None,
        max_epochs: Optional[int] = None,
        loss_weighting: Optional[str] = None,
        num_species_classes: Optional[int] = None,
        peft_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_url=weights_url,
            weights_path=weights_path,
        )
        # Optionally inject LoRA into the frozen backbone
        if peft_cfg and bool(peft_cfg.get("enabled", False)):
            try:
                # Allow gradient flow for adapters only
                feature_extractor.inference_only = False
            except Exception:
                pass
            feature_extractor, lora_cfg = inject_lora_into_feature_extractor(feature_extractor, peft_cfg)
            # Persist LoRA LR and WD for optimizer grouping
            self._peft_lora_lr: Optional[float] = float(peft_cfg.get("lora_lr", 5e-5))
            self._peft_lora_weight_decay: Optional[float] = float(peft_cfg.get("lora_weight_decay", 0.0))
        else:
            self._peft_lora_lr = None
            self._peft_lora_weight_decay = None
        if not freeze_backbone:
            for parameter in feature_extractor.backbone.parameters():
                parameter.requires_grad = True
            feature_extractor.backbone.train()
        self.feature_extractor = feature_extractor

        # Shared bottleneck: MLP defined by hidden_dims; last hidden dim is bottleneck size
        hidden_dims: List[int] = list(head_hidden_dims or [512, 256])
        act_name = head_activation
        layers: List[nn.Module] = []
        in_dim = embedding_dim
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        def _act():
            name = (act_name or "").lower()
            if name == "relu":
                return nn.ReLU(inplace=True)
            if name == "gelu":
                return nn.GELU()
            if name in ("silu", "swish"):
                return nn.SiLU(inplace=True)
            return nn.ReLU(inplace=True)
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(_act())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hd
        self.shared_bottleneck = nn.Sequential(*layers)

        # Task heads
        bottleneck_dim = hidden_dims[-1] if hidden_dims else embedding_dim
        self.reg_head = nn.Linear(bottleneck_dim, num_outputs)  # main 3-d regression
        self.height_head = nn.Linear(bottleneck_dim, 1)         # auxiliary regression
        if num_species_classes is None or int(num_species_classes) <= 1:
            raise ValueError("num_species_classes must be provided (>1) for classification task")
        self.num_species_classes = int(num_species_classes)
        self.species_head = nn.Linear(bottleneck_dim, self.num_species_classes)
        self.out_softplus = nn.Softplus() if use_output_softplus else None

        # Uncertainty Weighting (UW) parameters (optional)
        self.loss_weighting: Optional[str] = (loss_weighting.lower() if loss_weighting else None)
        if self.loss_weighting == "uw":
            # three tasks: reg3, height, species
            self.uw_logvars = nn.Parameter(torch.zeros(3))

        # buffers for epoch-wise metrics on validation set
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, images: Tensor) -> Tensor:
        # Return only main 3-d regression prediction (for compatibility and inference)
        features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        out = self.reg_head(z)
        if self.out_softplus is not None:
            out = self.out_softplus(out)
        return out

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        # batch is a dict from the dataset
        images: Tensor = batch["image"]
        y_reg3: Tensor = batch["y_reg3"]  # (B,3)
        y_height: Tensor = batch["y_height"]  # (B,1)
        y_species: Tensor = batch["y_species"]  # (B,)

        features = self.feature_extractor(images)
        z = self.shared_bottleneck(features)
        pred_reg3 = self.reg_head(z)
        if self.out_softplus is not None:
            pred_reg3 = self.out_softplus(pred_reg3)
        pred_height = self.height_head(z)
        logits_species = self.species_head(z)

        # Task losses
        loss_reg3 = F.mse_loss(pred_reg3, y_reg3)  # agg over dims
        loss_height = F.mse_loss(pred_height, y_height)
        loss_species = F.cross_entropy(logits_species, y_species)

        # Metrics logging
        # per-task losses
        self.log(f"{stage}_loss_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_loss_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_loss_species", loss_species, on_step=False, on_epoch=True, prog_bar=False)
        # per-task metrics
        self.log(f"{stage}_mse_reg3", loss_reg3, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse_height", loss_height, on_step=False, on_epoch=True, prog_bar=False)
        mae_reg3 = F.l1_loss(pred_reg3, y_reg3)
        mae_height = F.l1_loss(pred_height, y_height)
        self.log(f"{stage}_mae_reg3", mae_reg3, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mae_height", mae_height, on_step=False, on_epoch=True, prog_bar=False)
        # per-dimension MSE for the 3-d regression task
        with torch.no_grad():
            per_dim_mse = torch.mean((pred_reg3 - y_reg3) ** 2, dim=0)
            for i in range(per_dim_mse.shape[0]):
                self.log(f"{stage}_mse_reg3_{i}", per_dim_mse[i], on_step=False, on_epoch=True, prog_bar=False)
        with torch.no_grad():
            acc = (logits_species.argmax(dim=-1) == y_species).float().mean()
        self.log(f"{stage}_acc_species", acc, on_step=False, on_epoch=True, prog_bar=False)

        # UW over three tasks or equal weighting
        if self.loss_weighting == "uw":
            s = self.uw_logvars  # (3,)
            losses_vec = torch.stack([loss_reg3, loss_height, loss_species])
            total_loss = 0.5 * torch.sum(torch.exp(-s) * losses_vec + s)
            with torch.no_grad():
                w_eff = 0.5 * torch.exp(-s)
                sigma = torch.exp(0.5 * s)
                for i, name in enumerate(["reg3", "height", "species"]):
                    self.log(f"{stage}_uw_w_{name}", w_eff[i], on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"{stage}_uw_logvar_{name}", s[i], on_step=False, on_epoch=True, prog_bar=False)
                    self.log(f"{stage}_uw_sigma_{name}", sigma[i], on_step=False, on_epoch=True, prog_bar=False)
        else:
            total_loss = (loss_reg3 + loss_height + loss_species) / 3.0

        # overall metrics for backward-compat
        mae = F.l1_loss(pred_reg3, y_reg3)
        mse = F.mse_loss(pred_reg3, y_reg3)
        self.log(f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        return {
            "loss": total_loss,
            "mae": mae,
            "mse": mse,
            "preds": pred_reg3,
            "targets": y_reg3,
        }

    def training_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        out = self._shared_step(batch, stage="val")
        self._val_preds.append(out["preds"].detach().float().cpu())
        self._val_targets.append(out["targets"].detach().float().cpu())

    def on_validation_epoch_start(self) -> None:
        self._val_preds.clear()
        self._val_targets.clear()

    def on_validation_epoch_end(self) -> None:
        if len(self._val_preds) == 0:
            return
        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        # overall R^2 across all outputs
        ss_res = torch.sum((targets - preds) ** 2)
        mean_t = torch.mean(targets)
        ss_tot = torch.sum((targets - mean_t) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
        self.log("val_r2", r2, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Separate parameter groups: LoRA adapters (smaller LR) and the rest (head, optionally others)
        lora_params: List[torch.nn.Parameter] = []
        try:
            lora_params = get_lora_param_list(self.feature_extractor.backbone)
        except Exception:
            lora_params = []

        all_params = [p for p in self.parameters() if p.requires_grad]
        lora_set = set(lora_params)
        other_params = [p for p in all_params if p not in lora_set]

        param_groups: List[Dict[str, Any]] = []
        if len(other_params) > 0:
            param_groups.append({
                "params": other_params,
                "lr": self.hparams.learning_rate,
                "weight_decay": self.hparams.weight_decay,
            })
        if len(lora_params) > 0:
            lora_lr = float(self._peft_lora_lr or (self.hparams.learning_rate * 0.1))
            lora_wd = float(self._peft_lora_weight_decay if self._peft_lora_weight_decay is not None else self.hparams.weight_decay)
            param_groups.append({
                "params": lora_params,
                "lr": lora_lr,
                "weight_decay": lora_wd,
            })

        optimizer: Optimizer = AdamW(param_groups)

        if self.hparams.scheduler_name and self.hparams.scheduler_name.lower() == "cosine":
            max_epochs = self.hparams.max_epochs or 10
            scheduler: _LRScheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": "val_loss",
                },
            }
        return optimizer


