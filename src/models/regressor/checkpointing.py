from __future__ import annotations


class RegressorCheckpointingMixin:
    # --- Lightweight checkpointing: exclude frozen backbone weights ---
    def state_dict(self, destination=None, prefix: str = "", keep_vars: bool = False):  # type: ignore[override]
        # Get the full state dict from the parent
        full = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # If the backbone is frozen, strip its heavy weights while keeping LoRA params
        try:
            freeze_backbone: bool = bool(getattr(self.hparams, "freeze_backbone", True))
        except Exception:
            freeze_backbone = True
        if not freeze_backbone:
            return full
        filtered = type(full)()
        for key, value in full.items():
            # Keep LoRA adapter parameters inside the backbone
            if key.startswith(f"{prefix}feature_extractor.backbone"):
                if ("lora_" in key) or ("lora_magnitude_vector" in key):
                    filtered[key] = value
                # else: drop heavy backbone weights
                continue
            # Keep everything else (heads, bottleneck, buffers, optimizer hooks, etc.)
            filtered[key] = value
        return filtered

    def load_state_dict(self, state_dict: dict, strict: bool = True):  # type: ignore[override]
        """
        Load with tolerance for missing backbone weights (since we exclude them from ckpt).
        """
        try:
            # Always allow missing/unexpected keys to avoid failures due to stripped backbone
            return super().load_state_dict(state_dict, strict=False)
        except Exception:
            # Fallback to default behavior if anything goes wrong
            return super().load_state_dict(state_dict, strict=strict)


