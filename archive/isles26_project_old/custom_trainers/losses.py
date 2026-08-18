"""
Loss functions for the loss-function study (Phase 2).

These are written to slot into nnU-Net v2's `_build_loss` mechanism the same
way its built-in `DC_and_CE_loss` does: they take raw network logits and an
integer label map of matching spatial shape, and nnU-Net's own
`DeepSupervisionWrapper` takes care of applying the loss at every deep
supervision scale and weighting them -- we don't need to reimplement that
part, just the per-scale loss itself.

All losses assume the standard nnU-Net setup for this task: a single
foreground class (lesion) plus background, trained with softmax over 2
channels. If your `dataset.json` ends up with more than one foreground
label, `TverskyLoss` / `FocalTverskyLoss` below already generalize (per-class
Tversky averaged over foreground classes); `FocalLoss` also generalizes as
written.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """target: (B, 1, *spatial) integer labels -> (B, C, *spatial) one-hot."""
    target = target.long()
    shape = list(target.shape)
    shape[1] = num_classes
    one_hot = torch.zeros(shape, device=target.device, dtype=torch.float32)
    one_hot.scatter_(1, target, 1)
    return one_hot


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017), softmax formulation.

    Down-weights easy (already well-classified) voxels -- mostly background,
    given how small stroke lesions are -- so gradient signal concentrates on
    the hard, usually lesion-boundary, voxels.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # net_output: (B, C, *spatial) logits; target: (B, 1, *spatial) long
        if target.ndim == net_output.ndim:
            target = target[:, 0]
        else:
            target = target.long()
        mask = target != self.ignore_index
        log_probs = F.log_softmax(net_output, dim=1)
        probs = log_probs.exp()

        target_clamped = target.clamp(min=0)
        log_pt = log_probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target_clamped.unsqueeze(1)).squeeze(1)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_weight * log_pt
        loss = loss[mask]
        return loss.mean() if loss.numel() > 0 else loss.sum()


class TverskyLoss(nn.Module):
    """Tversky loss (Salehi et al., 2017): asymmetric generalization of Dice.

    alpha weights false positives, beta weights false negatives. For small
    structures like stroke lesions, beta > alpha (penalize missed lesion
    voxels more than over-segmentation) typically helps recall.
    Defaults below (alpha=0.3, beta=0.7) follow the common small-lesion
    recommendation; treat these as a tunable hyperparameter, not gospel.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def _per_class_tversky(self, probs: torch.Tensor, one_hot: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Sum over batch + spatial dims, keep class dim.
        dims = (0,) + tuple(range(2, probs.ndim))
        probs = probs * mask
        one_hot = one_hot * mask

        tp = (probs * one_hot).sum(dim=dims)
        fp = (probs * (1 - one_hot)).sum(dim=dims)
        fn = ((1 - probs) * one_hot).sum(dim=dims)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return tversky

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = net_output.shape[1]
        if target.ndim != net_output.ndim:
            target = target.unsqueeze(1)
        mask = (target != self.ignore_index).float()
        target_clamped = target.clamp(min=0)

        probs = F.softmax(net_output, dim=1)
        one_hot = _to_one_hot(target_clamped, num_classes)

        tversky_per_class = self._per_class_tversky(probs, one_hot, mask)
        # Average Tversky index over foreground classes only (skip background = class 0).
        fg_tversky = tversky_per_class[1:] if num_classes > 1 else tversky_per_class
        return 1.0 - fg_tversky.mean()


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky (Abraham & Khan, 2019): Tversky index raised to 1/gamma.

    Adds a focal-style term on top of Tversky so that easy cases (already
    high Tversky / near-correct lesions) contribute even less gradient,
    further concentrating learning on hard, small, or boundary-heavy cases.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 0.75, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.gamma = gamma
        self._tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth, ignore_index=ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_loss = self._tversky(net_output, target)  # = 1 - Tversky index
        return torch.pow(tversky_loss.clamp(min=1e-7), self.gamma)


class DiceOnlyLoss(nn.Module):
    """Plain soft Dice loss, no CE term -- the "no imbalance handling at all
    beyond standard Dice" control condition in the loss study."""

    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = net_output.shape[1]
        if target.ndim != net_output.ndim:
            target = target.unsqueeze(1)
        mask = (target != self.ignore_index).float()
        target_clamped = target.clamp(min=0)

        probs = F.softmax(net_output, dim=1) * mask
        one_hot = _to_one_hot(target_clamped, num_classes) * mask

        dims = (0,) + tuple(range(2, probs.ndim))
        intersection = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2 * intersection + self.smooth) / (denom + self.smooth)
        fg_dice = dice_per_class[1:] if num_classes > 1 else dice_per_class
        return 1.0 - fg_dice.mean()
