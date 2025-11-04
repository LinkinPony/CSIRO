from typing import Any, Dict, Optional, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from lightning.pytorch import LightningModule
from loguru import logger

from .backbone import build_feature_extractor
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
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name,
            pretrained=pretrained,
            weights_url=weights_url,
            weights_path=weights_path,
        )
        if not freeze_backbone:
            for parameter in feature_extractor.backbone.parameters():
                parameter.requires_grad = True
            feature_extractor.backbone.train()
        self.feature_extractor = feature_extractor

        self.head = build_head_layer(
            embedding_dim=embedding_dim,
            num_outputs=num_outputs,
            head_hidden_dims=head_hidden_dims,
            head_activation=head_activation,
            dropout=dropout,
            use_output_softplus=use_output_softplus,
        )

        # buffers for epoch-wise metrics on validation set
        self._val_preds: list[Tensor] = []
        self._val_targets: list[Tensor] = []

    def forward(self, images: Tensor) -> Tensor:
        features = self.feature_extractor(images)
        outputs = self.head(features)
        return outputs

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        images, targets = batch
        preds = self(images)
        loss = F.mse_loss(preds, targets)
        mae = F.l1_loss(preds, targets)
        mse = loss
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_mse", mse, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "mae": mae, "mse": mse, "preds": preds, "targets": targets}

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
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer: Optimizer = AdamW(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

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


