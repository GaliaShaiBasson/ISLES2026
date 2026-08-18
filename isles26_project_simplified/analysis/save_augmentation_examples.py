#!/usr/bin/env python3
"""Save before/after example slices of stock vs. widened (WideAug) intensity augmentation.

Expected input: an already-preprocessed nnU-Net case under
``nnUNet_preprocessed/<dataset>/nnUNetPlans_3d_fullres/`` (blosc2 ``.b2nd`` +
``.pkl``, nnU-Net 2.8.1's preprocessed format) -- run ``isles26.py preprocess``
first if it doesn't exist yet.

Produces: one PNG per pipeline (stock, wideaug) under ``--out-dir``, each a grid
of [original, augmented x N] axial slices through the lesion center, plus a
combined side-by-side comparison PNG.

Non-obvious rationale: PROJECT_PLAN.md flags this as time-sensitive to capture
while ``nnUNetTrainerWideAug`` is fresh -- nnU-Net does not save augmented
samples anywhere during training, so reproducing this later means re-deriving
the exact transform config from source rather than reading it off a saved
artifact. Both pipelines' ``get_training_transforms`` are ``@staticmethod``,
so this calls them directly with the same real values nnU-Net computed for
tonight's actual ``baseline-wideaug-500`` run (patch_size=(128,128,128),
rotation_for_DA=+-30 deg, mirror_axes=(0,1,2), do_dummy_2d_data_aug=False --
see its ``training_log_*.txt``) rather than reinstantiating a full trainer.

Usage:
    python analysis/save_augmentation_examples.py \
        --case-id ATLAS_r001s001_ses1 --num-samples 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _load_case(preprocessed_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

    dataset = nnUNetDatasetBlosc2(str(preprocessed_dir), identifiers=[case_id])
    data, seg, _, _ = dataset.load_case(case_id)
    return np.asarray(data), np.asarray(seg)


def _center_crop_or_pad(volume: np.ndarray, center_zyx: tuple[int, int, int], patch_size: tuple[int, int, int]) -> np.ndarray:
    """volume: (C, Z, Y, X). Returns a (C, *patch_size) crop centered on center_zyx, zero-padded if needed."""
    channels = volume.shape[0]
    out = np.zeros((channels, *patch_size), dtype=volume.dtype)
    starts_src, starts_dst, lengths = [], [], []
    for center, size, vol_size in zip(center_zyx, patch_size, volume.shape[1:]):
        lo = center - size // 2
        hi = lo + size
        src_lo, dst_lo = max(lo, 0), max(-lo, 0)
        src_hi = min(hi, vol_size)
        length = src_hi - src_lo
        starts_src.append(src_lo)
        starts_dst.append(dst_lo)
        lengths.append(max(length, 0))
    src_slices = tuple(slice(s, s + l) for s, l in zip(starts_src, lengths))
    dst_slices = tuple(slice(s, s + l) for s, l in zip(starts_dst, lengths))
    out[(slice(None), *dst_slices)] = volume[(slice(None), *src_slices)]
    return out


def _lesion_center(seg: np.ndarray) -> tuple[int, int, int]:
    foreground = np.argwhere(seg[0] > 0)
    if foreground.size == 0:
        return tuple(s // 2 for s in seg.shape[1:])
    return tuple(int(round(v)) for v in foreground.mean(axis=0))


def _crop_bounds(center_zyx: tuple[int, int, int], patch_size: tuple[int, int, int], vol_shape: tuple[int, int, int]):
    """Actual (clipped) [lo, hi) bounds per axis used by _center_crop_or_pad, for drawing a context box."""
    bounds = []
    for center, size, vol_size in zip(center_zyx, patch_size, vol_shape):
        lo = max(center - size // 2, 0)
        hi = min(lo + size, vol_size)
        bounds.append((lo, hi))
    return bounds


# Same numeric ranges nnUNetTrainerWideAug.py documents/uses -- kept here (not imported)
# since these are display-only constants for the *isolated, probability-forced-to-1*
# illustration below, deliberately not the real stochastic pipeline (see module docstring
# on why get_training_transforms's real ~15-30% activation rate makes a single random
# draw a poor illustration of the range difference).
_STOCK_BRIGHTNESS_CONTRAST_RANGE = (0.75, 1.25)
_WIDE_BRIGHTNESS_CONTRAST_RANGE = (0.6, 1.4)
_STOCK_GAMMA_RANGE = (0.7, 1.5)
_WIDE_GAMMA_RANGE = (0.6, 1.6)


def _forced_intensity_transform(brightness_contrast_range, gamma_range):
    """Brightness -> Contrast -> Gamma, each ALWAYS applied (no RandomTransform probability
    wrapper), so every draw visibly shows the configured range -- unlike the real training
    pipeline where each only fires 15-30% of the time. For illustration only; not what
    training actually runs (see ``_build_transforms`` for that)."""
    from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
    from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
    from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
    from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

    return ComposeTransforms([
        MultiplicativeBrightnessTransform(
            multiplier_range=BGContrast(brightness_contrast_range), synchronize_channels=False, p_per_channel=1
        ),
        ContrastTransform(
            contrast_range=BGContrast(brightness_contrast_range), preserve_range=True,
            synchronize_channels=False, p_per_channel=1
        ),
        GammaTransform(
            gamma=BGContrast(gamma_range), p_invert_image=0, synchronize_channels=False,
            p_per_channel=1, p_retain_stats=1
        ),
    ])


def _build_transforms(cls, patch_size: tuple[int, int, int]):
    rotation_for_da = (-30.0 / 360 * 2 * np.pi, 30.0 / 360 * 2 * np.pi)
    return cls.get_training_transforms(
        patch_size=patch_size,
        rotation_for_DA=rotation_for_da,
        deep_supervision_scales=None,
        mirror_axes=(0, 1, 2),
        do_dummy_2d_data_aug=False,
        use_mask_for_norm=None,
        is_cascaded=False,
    )


def _apply(transform, image: np.ndarray, seg: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    sample = {"image": torch.from_numpy(image.copy()).float(), "segmentation": torch.from_numpy(seg.copy()).float()}
    out = transform(**sample)
    return out["image"].numpy()


def _display_range(slice_2d: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(slice_2d, [1, 99])
    return (float(lo), float(hi)) if hi > lo else (0.0, 1.0)


def _clip_for_display(slice_2d: np.ndarray, display_range: tuple[float, float]) -> np.ndarray:
    # Deliberately NOT re-normalized per panel: brightness/contrast/gamma augmentation
    # changes intensity values, and per-panel percentile stretching would silently
    # rescale that difference away, defeating the point of this figure. Every panel is
    # clipped to the SAME range (from the original slice), so a visibly brighter/
    # darker/higher-contrast panel reflects a real intensity change, not a display
    # artifact.
    lo, hi = display_range
    return np.clip((slice_2d - lo) / (hi - lo), 0, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-id", default="ATLAS_r001s001_ses1", help="case with a large, easy-to-see lesion by default")
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT.parents[0] / "nnUNet_preprocessed" / "Dataset002_ATLAS" / "nnUNetPlans_3d_fullres"))
    parser.add_argument("--patch-size", default="128,128,128")
    parser.add_argument("--num-samples", type=int, default=4, help="augmented draws per pipeline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="workspace/figures/augmentation_examples")
    args = parser.parse_args()

    patch_size = tuple(int(x) for x in args.patch_size.split(","))
    preprocessed_dir = Path(args.dataset_dir)
    if not preprocessed_dir.is_dir():
        parser.error(f"Preprocessed dataset dir not found: {preprocessed_dir}")

    data, seg = _load_case(preprocessed_dir, args.case_id)
    center = _lesion_center(seg)
    image_crop = _center_crop_or_pad(data, center, patch_size)
    seg_crop = _center_crop_or_pad(seg, center, patch_size)
    mid_z = patch_size[0] // 2

    from custom_trainers.nnUNetTrainerWideAug import nnUNetTrainerWideAug
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    pipelines = {"stock": nnUNetTrainer, "wideaug": nnUNetTrainerWideAug}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_original_slice = image_crop[0, mid_z]
    display_range = _display_range(raw_original_slice)

    panels = {}
    for name, cls in pipelines.items():
        transform = _build_transforms(cls, patch_size)
        augmented_slices = []
        for draw in range(args.num_samples):
            augmented = _apply(transform, image_crop, seg_crop, seed=args.seed + draw)
            augmented_slices.append(_clip_for_display(augmented[0, mid_z], display_range))
        panels[name] = augmented_slices

    original_slice = _clip_for_display(raw_original_slice, display_range)

    n_cols = args.num_samples + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    for row, (name, slices) in enumerate(panels.items()):
        axes[row, 0].imshow(original_slice, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{name}: original")
        axes[row, 0].axis("off")
        for col, sl in enumerate(slices, start=1):
            axes[row, col].imshow(sl, cmap="gray", vmin=0, vmax=1)
            axes[row, col].set_title(f"{name}: sample {col}")
            axes[row, col].axis("off")
    fig.suptitle(
        f"As seen during real training -- case {args.case_id}, axial slice {mid_z}\n"
        "(each intensity transform only fires ~15-30% of the time; a difference not showing here is expected)"
    )
    fig.tight_layout()
    out_path = out_dir / f"augmentation_comparison_{args.case_id}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # --- Forced-intensity illustration: probability=1, so the range difference is
    # actually visible (not what training runs -- see module docstring). ---
    forced_transforms = {
        "stock": _forced_intensity_transform(_STOCK_BRIGHTNESS_CONTRAST_RANGE, _STOCK_GAMMA_RANGE),
        "wideaug": _forced_intensity_transform(_WIDE_BRIGHTNESS_CONTRAST_RANGE, _WIDE_GAMMA_RANGE),
    }
    forced_panels = {}
    for name, transform in forced_transforms.items():
        forced_panels[name] = [
            _clip_for_display(_apply(transform, image_crop, seg_crop, seed=args.seed + draw)[0, mid_z], display_range)
            for draw in range(args.num_samples)
        ]
    fig2, axes2 = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    for row, (name, slices) in enumerate(forced_panels.items()):
        axes2[row, 0].imshow(original_slice, cmap="gray", vmin=0, vmax=1)
        axes2[row, 0].set_title(f"{name}: original")
        axes2[row, 0].axis("off")
        for col, sl in enumerate(slices, start=1):
            axes2[row, col].imshow(sl, cmap="gray", vmin=0, vmax=1)
            axes2[row, col].set_title(f"{name}: sample {col}")
            axes2[row, col].axis("off")
    fig2.suptitle(
        f"Isolated brightness/contrast/gamma range, probability forced to 1 -- case {args.case_id}\n"
        f"stock: x{_STOCK_BRIGHTNESS_CONTRAST_RANGE}/gamma{_STOCK_GAMMA_RANGE}   "
        f"wideaug: x{_WIDE_BRIGHTNESS_CONTRAST_RANGE}/gamma{_WIDE_GAMMA_RANGE}   (illustration only, not the real pipeline)"
    )
    fig2.tight_layout()
    forced_path = out_dir / f"augmentation_forced_intensity_{args.case_id}.png"
    fig2.savefig(forced_path, dpi=150)
    plt.close(fig2)
    print(f"Wrote {forced_path}")

    # --- Full-volume context: the whole head, with the training-patch region boxed. ---
    full_slice = _clip_for_display(data[0, center[0]], display_range)
    bounds = _crop_bounds(center, patch_size, data.shape[1:])
    (_, _), (y_lo, y_hi), (x_lo, x_hi) = bounds
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))
    ax3[0].imshow(full_slice, cmap="gray", vmin=0, vmax=1)
    ax3[0].add_patch(plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo, edgecolor="red", facecolor="none", linewidth=1.5))
    ax3[0].set_title(f"Full preprocessed volume ({data.shape[1]}x{data.shape[2]}x{data.shape[3]})\nred box = training patch region")
    ax3[0].axis("off")
    ax3[1].imshow(original_slice, cmap="gray", vmin=0, vmax=1)
    ax3[1].set_title(f"Training patch only ({patch_size[0]}x{patch_size[1]}x{patch_size[2]})")
    ax3[1].axis("off")
    fig3.suptitle(f"Whole-scan context vs. actual network input -- case {args.case_id}, axial slice {center[0]}")
    fig3.tight_layout()
    context_path = out_dir / f"augmentation_context_{args.case_id}.png"
    fig3.savefig(context_path, dpi=150)
    plt.close(fig3)
    print(f"Wrote {context_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
