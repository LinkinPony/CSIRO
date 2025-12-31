"""
Test-time / transductive post-training utilities.

This package implements A4-style adaptation:
  - train LoRA + regression head on unlabeled images (typically the test set)
  - keep the frozen DINOv3 backbone weights unchanged (only LoRA adapters are trainable)
  - export an updated head package in the same format as HeadCheckpoint:
      {"state_dict": ..., "meta": ..., "peft": {...}}.
"""


