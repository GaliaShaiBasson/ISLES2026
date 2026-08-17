"""Losses used by the nnU-Net trainer variants."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_target(target: torch.Tensor, net_output: torch.Tensor, ignore_index: int):
    if target.ndim == net_output.ndim:
        target = target[:, 0]
    target = target.long()
    valid = target != ignore_index
    safe = torch.where(valid, target, torch.zeros_like(target))
    if torch.any((safe < 0) | (safe >= net_output.shape[1])):
        raise ValueError("Target contains a class index outside the network output range")
    return safe, valid


def _to_one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), num_classes=num_classes).movedim(-1, 1).float()


class FocalLoss(nn.Module):
    """Softmax focal loss with class-specific alpha for binary segmentation."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, ignore_index: int = -100):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.ignore_index = int(ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        safe_target, valid = _prepare_target(target, net_output, self.ignore_index)
        log_probs = F.log_softmax(net_output, dim=1)
        probs = log_probs.exp()
        gather_index = safe_target.unsqueeze(1)
        log_pt = log_probs.gather(1, gather_index).squeeze(1)
        pt = probs.gather(1, gather_index).squeeze(1)

        if net_output.shape[1] == 2:
            alpha_t = torch.where(
                safe_target == 1,
                torch.as_tensor(self.alpha, device=net_output.device, dtype=net_output.dtype),
                torch.as_tensor(1 - self.alpha, device=net_output.device, dtype=net_output.dtype),
            )
        else:
            alpha_t = torch.ones_like(pt)

        loss = -alpha_t * (1 - pt).pow(self.gamma) * log_pt
        loss = loss[valid]
        return loss.mean() if loss.numel() else net_output.sum() * 0


class TverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        safe_target, valid = _prepare_target(target, net_output, self.ignore_index)
        probs = F.softmax(net_output, dim=1)
        one_hot = _to_one_hot(safe_target, net_output.shape[1])
        mask = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0,) + tuple(range(2, probs.ndim))
        true_positive = (probs * one_hot).sum(dim=dims)
        false_positive = (probs * (1 - one_hot)).sum(dim=dims)
        false_negative = ((1 - probs) * one_hot).sum(dim=dims)
        score = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        foreground = score[1:] if net_output.shape[1] > 1 else score
        return 1 - foreground.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.tversky = TverskyLoss(alpha, beta, smooth, ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.tversky(net_output, target).clamp(min=1e-7).pow(self.gamma)


class DiceTverskyLoss(nn.Module):
    """Compound loss: dice_weight*Dice + tversky_weight*Tversky (unweighted sum by default,
    mirrors nnU-Net's own DC_and_CE_loss pattern of summing rather than averaging terms).

    Dice keeps the region-overlap gradient that a standalone Tversky/Focal term lacks a
    counterweight for -- see the 2026-08-18 CLAUDE.md entry: plain nnUNetTrainerTversky
    (alpha=0.3/beta=0.7, no Dice term) genuinely improved small-lesion recall (best
    true-positive/false-negative rate of every condition tested) but at the cost of enough
    extra false-positive components that lesion-wise F1 still landed below baseline.
    Softening the asymmetry (alpha=0.4/beta=0.6) and adding Dice back in is meant to keep
    the recall benefit while reining in that false-positive cost.
    """

    def __init__(
        self,
        tversky_alpha: float = 0.4,
        tversky_beta: float = 0.6,
        dice_weight: float = 1.0,
        tversky_weight: float = 1.0,
        smooth: float = 1e-5,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.dice = DiceOnlyLoss(smooth=smooth, ignore_index=ignore_index)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta, smooth=smooth, ignore_index=ignore_index)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(net_output, target) + self.tversky_weight * self.tversky(net_output, target)


class DiceOnlyLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        safe_target, valid = _prepare_target(target, net_output, self.ignore_index)
        probs = F.softmax(net_output, dim=1)
        one_hot = _to_one_hot(safe_target, net_output.shape[1])
        mask = valid.unsqueeze(1).to(probs.dtype)
        probs = probs * mask
        one_hot = one_hot * mask

        dims = (0,) + tuple(range(2, probs.ndim))
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        foreground = score[1:] if net_output.shape[1] > 1 else score
        return 1 - foreground.mean()
