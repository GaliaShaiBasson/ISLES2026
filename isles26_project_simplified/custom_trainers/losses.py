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
