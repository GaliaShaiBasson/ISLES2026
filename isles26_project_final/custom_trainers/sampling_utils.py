"""Pure helpers for lesion-volume-aware case sampling.

These functions deliberately have no nnU-Net dependency so they can be unit-tested
without a GPU or an nnU-Net installation.
"""
from __future__ import annotations

import numpy as np


def inverse_volume_probabilities(volumes_mm3, power: float = 1.0, floor_mm3: float = 1.0) -> np.ndarray:
    """Return normalized probabilities proportional to ``volume ** -power``.

    ``power=0`` is uniform sampling, ``power=0.5`` is a mild small-lesion bias,
    and ``power=1`` is the fixed inverse-volume sampler used in the main method.
    """
    if power < 0:
        raise ValueError("power must be non-negative")
    if floor_mm3 <= 0:
        raise ValueError("floor_mm3 must be positive")

    values = np.asarray(volumes_mm3, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("volumes_mm3 must be a non-empty 1D sequence")
    if not np.isfinite(values).all():
        raise ValueError("volumes_mm3 must contain only finite values")
    if (values < 0).any():
        raise ValueError("lesion volumes cannot be negative")

    safe = np.maximum(values, float(floor_mm3))
    raw = np.ones_like(safe) if power == 0 else np.power(safe, -float(power))
    total = float(raw.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("could not derive finite sampling probabilities")
    return raw / total


def curriculum_power(epoch: int, num_epochs: int) -> tuple[str, float]:
    """Three-phase lesion-size curriculum.

    The schedule is defined by fractions of the actual training length, so it also
    works for shortened debug schedules:

    - first third: uniform sampling (power 0.0)
    - middle third: mild inverse-sqrt volume bias (power 0.5)
    - final third: full inverse-volume bias (power 1.0)
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    epoch = max(0, int(epoch))
    progress = min(epoch / float(num_epochs), 1.0)
    if progress < 1.0 / 3.0:
        return "uniform", 0.0
    if progress < 2.0 / 3.0:
        return "mild", 0.5
    return "full", 1.0
