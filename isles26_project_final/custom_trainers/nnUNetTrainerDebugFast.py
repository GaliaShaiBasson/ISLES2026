"""Fast smoke-test trainer for nnU-Net 2.8.1.

This is intentionally unsuitable for scientific results. It only validates
loading, augmentation, optimization, validation, and checkpoint writing.
"""
from __future__ import annotations

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

NUM_EPOCHS = 5
ITERS_PER_EPOCH = 20


class nnUNetTrainerDebugMixin:
    """Reduce the training schedule while preserving nnU-Net's constructor."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = NUM_EPOCHS
        self.num_iterations_per_epoch = ITERS_PER_EPOCH
        self.num_val_iterations_per_epoch = min(self.num_val_iterations_per_epoch, 10)
        self.print_to_log_file(
            f"[DEBUG] {self.num_epochs} epochs x {self.num_iterations_per_epoch} iterations. "
            "Do not use this checkpoint for reported results."
        )


class nnUNetTrainerDebugFast(nnUNetTrainerDebugMixin, nnUNetTrainer):
    pass
