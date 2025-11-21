from __future__ import annotations

from typing import Dict, Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def _build_activation(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    # Fallback
    return nn.ReLU(inplace=True)


class ExpertMLP(nn.Module):
    """
    Simple 2-layer MLP expert used inside the PLE-style head.

    in_dim -> hidden_dim -> bottleneck_dim
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = _build_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, bottleneck_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TaskGatedExperts(nn.Module):
    """
    Progressive Layered Extraction-style gated experts for three tasks:
    reg3, NDVI, ratio. Each task has a private expert plus a shared expert.

    For a given task t in {reg3, ndvi, ratio}, its representation is:
        h_t = alpha_t0 * E_shared(x) + alpha_t1 * E_t(x),
    where alpha_t = softmax(W_t x) in R^2.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        expert_hidden_dim: int,
        activation: str = "gelu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Shared expert
        self.shared_expert = ExpertMLP(
            in_dim=in_dim,
            hidden_dim=expert_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout=dropout,
        )
        # Task-specific experts
        self.reg3_expert = ExpertMLP(
            in_dim=in_dim,
            hidden_dim=expert_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout=dropout,
        )
        self.ndvi_expert = ExpertMLP(
            in_dim=in_dim,
            hidden_dim=expert_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout=dropout,
        )
        self.ratio_expert = ExpertMLP(
            in_dim=in_dim,
            hidden_dim=expert_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            dropout=dropout,
        )

        # One 2-way gate per task: [shared, private]
        self.reg3_gate = nn.Linear(in_dim, 2)
        self.ndvi_gate = nn.Linear(in_dim, 2)
        self.ratio_gate = nn.Linear(in_dim, 2)

    def _mix(self, shared: Tensor, private: Tensor, gate_logits: Tensor) -> Tensor:
        """
        shared/private: (B, D)
        gate_logits: (B, 2)
        returns: (B, D)
        """
        alpha = F.softmax(gate_logits, dim=-1)  # (B,2)
        a_shared = alpha[:, 0:1]
        a_private = alpha[:, 1:2]
        return a_shared * shared + a_private * private

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            x: (B, in_dim) features from backbone (CLS || mean(patch)).

        Returns:
            dict with task-specific bottleneck representations:
              - "reg3":  (B, bottleneck_dim)
              - "ndvi":  (B, bottleneck_dim)
              - "ratio": (B, bottleneck_dim)
              - "shared": (B, bottleneck_dim)
        """
        shared = self.shared_expert(x)
        reg3_p = self.reg3_expert(x)
        ndvi_p = self.ndvi_expert(x)
        ratio_p = self.ratio_expert(x)

        h_reg3 = self._mix(shared, reg3_p, self.reg3_gate(x))
        h_ndvi = self._mix(shared, ndvi_p, self.ndvi_gate(x))
        h_ratio = self._mix(shared, ratio_p, self.ratio_gate(x))

        return {
            "reg3": h_reg3,
            "ndvi": h_ndvi,
            "ratio": h_ratio,
            "shared": shared,
        }


class PLEHead(nn.Module):
    """
    PLE-style multi-task regression head used both for training and offline inference.

    - Input: backbone features of size 2 * embedding_dim (CLS || mean(patch)).
    - Internally: TaskGatedExperts produce separate bottleneck reps for reg3/NDVI/ratio.
    - Outputs:
        * forward_multi(): dict with:
              - "reg3_logits": (B, num_outputs_main)
              - "ndvi":        (B, 1)            [if enable_ndvi]
              - "ratio_logits":(B, num_ratio_outputs) [if enable_ratio_head]
              - "z_reg3", "z_ndvi", "z_ratio": bottleneck reps
        * forward(): Tensor of shape (B, num_outputs_main + num_ratio_outputs),
          concatenating main regression outputs and ratio logits for inference.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        bottleneck_dim: int,
        num_outputs_main: int,
        num_ratio_outputs: int = 0,
        activation: str = "gelu",
        dropout: float = 0.0,
        expert_hidden_dim: Optional[int] = None,
        enable_ndvi: bool = False,
    ) -> None:
        super().__init__()
        if num_outputs_main < 1:
            raise ValueError("num_outputs_main must be >= 1")

        self.embedding_dim = int(embedding_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_outputs_main = int(num_outputs_main)
        self.num_ratio_outputs = int(max(0, num_ratio_outputs))
        self.enable_ndvi = bool(enable_ndvi)

        in_dim = self.embedding_dim * 2
        if expert_hidden_dim is None or expert_hidden_dim <= 0:
            expert_hidden_dim = max(1, self.bottleneck_dim // 4)
        self.expert_hidden_dim = int(expert_hidden_dim)

        # Gated experts
        self.gated_experts = TaskGatedExperts(
            in_dim=in_dim,
            bottleneck_dim=self.bottleneck_dim,
            expert_hidden_dim=self.expert_hidden_dim,
            activation=activation,
            dropout=dropout,
        )

        # Task heads
        self.reg3_heads = nn.ModuleList(
            [nn.Linear(self.bottleneck_dim, 1) for _ in range(self.num_outputs_main)]
        )
        self.ndvi_head: Optional[nn.Linear]
        if self.enable_ndvi:
            self.ndvi_head = nn.Linear(self.bottleneck_dim, 1)
        else:
            self.ndvi_head = None

        if self.num_ratio_outputs > 0:
            self.ratio_head: Optional[nn.Linear] = nn.Linear(
                self.bottleneck_dim, self.num_ratio_outputs
            )
        else:
            self.ratio_head = None

    def _compute_task_outputs(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Core computation shared by training and inference.
        """
        reps = self.gated_experts(x)
        h_reg3 = reps["reg3"]
        logits_list: List[Tensor] = []
        for head in self.reg3_heads:
            logits_list.append(head(h_reg3))
        reg3_logits = torch.cat(logits_list, dim=-1)  # (B, num_outputs_main)

        out: Dict[str, Tensor] = {
            "reg3_logits": reg3_logits,
            "z_reg3": h_reg3,
        }

        if self.enable_ndvi and self.ndvi_head is not None:
            h_ndvi = reps["ndvi"]
            out["ndvi"] = self.ndvi_head(h_ndvi)
            out["z_ndvi"] = h_ndvi

        if self.ratio_head is not None:
            h_ratio = reps["ratio"]
            out["ratio_logits"] = self.ratio_head(h_ratio)
            out["z_ratio"] = h_ratio

        return out

    def forward_multi(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Multi-task forward used during training.

        Returns a dict with logits and bottleneck representations per task.
        """
        return self._compute_task_outputs(x)

    def forward(self, x: Tensor) -> Tensor:
        """
        Inference forward used by offline scripts. Returns a single tensor:

          - If num_ratio_outputs == 0:
                (B, num_outputs_main)
          - Else:
                (B, num_outputs_main + num_ratio_outputs),
            where the first num_outputs_main dims are main regression outputs
            (e.g., Dry_Total_g / g/m^2) and the remaining dims are ratio logits.
        """
        out = self._compute_task_outputs(x)
        main = out["reg3_logits"]
        if self.ratio_head is None or self.num_ratio_outputs <= 0:
            return main
        ratio_logits = out["ratio_logits"]
        return torch.cat([main, ratio_logits], dim=-1)



