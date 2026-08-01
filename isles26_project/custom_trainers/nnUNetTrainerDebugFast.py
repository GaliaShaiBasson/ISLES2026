"""
CPU-friendly debug trainer(s) -- for verifying the pipeline runs end-to-end
(data loading, augmentation, loss computation, checkpointing) in minutes,
NOT for producing usable models. Real experiments need a GPU; see the note
in README.md.

Usage (combine with `2d` config for a real shot at running on CPU):

    nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerDebugFast

This trains for 5 epochs of 20 iterations each (100 total iterations)
instead of the default 1000 x 250 = 250,000. Adjust `NUM_EPOCHS` /
`ITERS_PER_EPOCH` below if you want a slightly longer smoke test.

You can also stack this with a loss variant by editing the base class it
inherits from, e.g.:

    class nnUNetTrainerDebugFastFocalTversky(nnUNetTrainerDebugMixin, nnUNetTrainerFocalTversky):
        pass

(mixin must come first in the MRO so its epoch/iteration overrides win).
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

NUM_EPOCHS = 5
ITERS_PER_EPOCH = 20


class nnUNetTrainerDebugMixin:
    """Mixin overriding epoch length/count for fast smoke tests. Put this
    first in the MRO of any combined debug+variant trainer.

    IMPORTANT: nnU-Net's own `nnUNetTrainer.__init__` inspects
    `inspect.signature(self.__init__)` to record init kwargs for
    checkpointing/resuming. That resolves to the most-derived `__init__` in
    the MRO -- so this method MUST declare the same named parameters as
    `nnUNetTrainer.__init__` (not a generic `*args, **kwargs`), or that
    introspection breaks with a KeyError at construction time.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = NUM_EPOCHS
        self.num_iterations_per_epoch = ITERS_PER_EPOCH
        # Validation can stay cheap too -- a handful of batches is enough
        # to confirm the val loop and metric logging work.
        self.num_val_iterations_per_epoch = min(getattr(self, "num_val_iterations_per_epoch", 50), 10)
        self.print_to_log_file(
            f"[nnUNetTrainerDebugMixin] DEBUG MODE: {self.num_epochs} epochs x "
            f"{self.num_iterations_per_epoch} iterations. Results from this run "
            f"are NOT meaningful -- this only verifies the pipeline runs."
        )


class nnUNetTrainerDebugFast(nnUNetTrainerDebugMixin, nnUNetTrainer):
    """Plain default loss (Dice+CE), fast schedule. Use this first, on `2d`
    config, to confirm your whole setup (data prep -> preprocessing ->
    training -> inference -> evaluation) works before touching a GPU."""
    pass
