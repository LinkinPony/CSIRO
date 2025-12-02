from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """
    Sharpness-Aware Minimization (SAM) optimizer wrapper.

    This implements the two-step SAM update described in:
        "Sharpness-Aware Minimization for Efficiently Improving Generalization"
    and is intended to wrap a base optimizer such as AdamW.
    """

    def __init__(
        self,
        params: Iterable,
        base_optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        # Store configuration and initialize as a generic optimizer.
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Build the actual inner optimizer on top of our param groups.
        self.base_optimizer: Optimizer = base_optimizer(self.param_groups, **kwargs)
        # Keep param_groups and defaults in sync with the inner optimizer.
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """
        Move parameters to the local maximizer w + e(w).
        """
        grad_norm = self._grad_norm()
        if grad_norm == 0.0:
            return

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive: bool = bool(group.get("adaptive", False))

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save current parameters for the second step.
                self.state[p]["old_p"] = p.data.clone()
                if adaptive:
                    e_w = (p.abs() * p.grad) * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                p.add_(e_w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """
        Restore parameters to w and apply the base optimizer update.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_p = self.state[p].get("old_p", None)
                if old_p is not None:
                    p.data = old_p

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None) -> Optional[Union[torch.Tensor, float]]:
        """
        SAM requires a closure that performs a full forward and backward pass.

        The typical usage pattern is:
            loss = closure()
            optimizer.first_step(zero_grad=True)
            closure()
            optimizer.second_step(zero_grad=True)

        This method mirrors the reference implementation and can be used with
        optimizers that expect a closure argument.
        """
        if closure is None:
            raise RuntimeError("SAM optimizer requires a closure to be provided.")

        closure = torch.enable_grad()(closure)

        self.first_step(zero_grad=True)
        loss = closure()
        self.second_step(zero_grad=True)
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """
        Compute the global L2 norm of gradients across all parameter groups.
        """
        # If there are no parameters, default to CPU tensor to avoid index errors.
        if len(self.param_groups) == 0 or len(self.param_groups[0]["params"]) == 0:
            return torch.tensor(0.0)

        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            adaptive: bool = bool(group.get("adaptive", False))
            for p in group["params"]:
                if p.grad is None:
                    continue
                if adaptive:
                    grad = (p.abs() * p.grad).detach()
                else:
                    grad = p.grad.detach()
                norms.append(grad.norm(p=2).to(shared_device))

        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):  # type: ignore[override]
        """
        Ensure the wrapped optimizer's param_groups stay in sync when loading.
        """
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


