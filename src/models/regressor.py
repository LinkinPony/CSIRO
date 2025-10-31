from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from lightning.pytorch import LightningModule
from loguru import logger

from .backbone import build_feature_extractor


class BiomassRegressor(LightningModule):
    def __init__(
        self,
        backbone_name: str,
        embedding_dim: int,
        num_outputs: int = 3,
        dropout: float = 0.0,
        pretrained: bool = True,
        weights_url: Optional[str] = None,
        freeze_backbone: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_name: Optional[str] = None,
        max_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        feature_extractor = build_feature_extractor(
            backbone_name=backbone_name, pretrained=pretrained, weights_url=weights_url
        )
        if not freeze_backbone:
            for parameter in feature_extractor.backbone.parameters():
                parameter.requires_grad = True
            feature_extractor.backbone.train()
        self.feature_extractor = feature_extractor

        layers = []
        if dropout and dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(embedding_dim, num_outputs))
        self.head = nn.Sequential(*layers)

    def forward(self, images: Tensor) -> Tensor:
        features = self.feature_extractor(images)
        outputs = self.head(features)
        return outputs

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, Tensor]:
        images, targets = batch
        preds = self(images)
        loss = F.mse_loss(preds, targets)
        mae = F.l1_loss(preds, targets)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss, "mae": mae}

    def training_step(self, batch: Any, batch_idx: int):
        return self._shared_step(batch, stage="train")["loss"]

    def validation_step(self, batch: Any, batch_idx: int):
        self._shared_step(batch, stage="val")

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


