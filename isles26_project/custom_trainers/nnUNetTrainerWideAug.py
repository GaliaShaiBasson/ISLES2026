"""Widened-intensity-augmentation trainer, for the cross-center generalization question.

**What it does.** Overrides ``get_training_transforms`` with the same transform
pipeline nnU-Net 2.8.1 ships (see ``nnUNetTrainer.get_training_transforms``),
changing only intensity augmentation strength. Spatial augmentation (elastic
deformation in particular) is deliberately left untouched -- see rationale
below. Loss, sampling, network, and everything else are untouched -- combine
this with an epoch-budget mixin the same way every other trainer in this
project does (see ``nnUNetTrainerWideAugBaseline_500epochs`` below), not
standalone.

**Non-obvious rationale.** CLAUDE.md (2026-08-17 "Cross-center generalization"
future-considerations entry) found that every intensity-augmentation range this
project uses is nnU-Net's untouched generic default -- brightness/contrast
x0.75-1.25, gamma 0.7-1.5, each applied at only 10-30% probability, elastic
deformation disabled entirely (``p_elastic_deform=0``) -- chosen for no reason
specific to this dataset's actual cross-scanner-generalization goal. That entry
proposed widening these ranges as the cheap first lever to try before anything
adversarial (domain-adversarial training), and PROJECT_PLAN.md / PROJECT_REVIEW.md
now schedule it as ``baseline-wideaug-500``, built on the Dice+CE baseline (not
the best-performing loss/sampling condition) specifically so
``baseline-500`` vs. this trainer is a clean single-variable A/B on the
``test_id``/``test_ood`` Dice gap, uncorrelated with any loss or sampling choice.

Concrete deltas from stock (see inline comments at each changed transform for
the exact stock value being replaced):
  - brightness/contrast: range widened 0.75-1.25 -> 0.6-1.4, probability
    raised 0.15 -> 0.3 (still well short of the "large rotation/intensity
    jitter can break anatomical/HU validity" line the training-tips deck
    warns about -- MRI intensities have no Hounsfield-unit-style absolute
    meaning to preserve, so a wider multiplicative/contrast range stays
    anatomically valid here).
  - gamma: range widened 0.7-1.5 -> 0.6-1.6, probability raised 0.1/0.3 ->
    0.2/0.4 for the invert/no-invert variants respectively.
  - elastic deformation: left OFF, same as stock (``p_elastic_deform=0``).
    Deliberately not enabled -- the training-tips deck warns explicitly that
    elastic deformation "may distort the pathology itself," and a stroke
    lesion's shape/extent is exactly the thing this project's Dice/lesion-F1
    metrics score. Spatial distortion of the lesion is a different, riskier
    kind of augmentation than intensity widening (which only touches how the
    same anatomy is *rendered*, not its shape) and isn't part of this
    trainer's question -- see PROJECT_REVIEW.md.
  - noise, blur, low-resolution simulation, scaling, rotation, mirroring,
    mask-for-norm, cascade, and deep-supervision-downsampling transforms:
    unchanged from stock, byte-for-byte.

Usage: ``python isles26.py train baseline-wideaug-500 --dataset-id 2``
"""
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import (
    RemoveRandomConnectedComponentFromOneHotEncodingTransform,
)
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWideAug(nnUNetTrainer):
    """Stock nnU-Net augmentation pipeline with widened intensity ranges + elastic deform."""

    @staticmethod
    def get_training_transforms(
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False,
                # p_elastic_deform=0: unchanged from stock, deliberately -- see module
                # docstring (elastic deformation risks distorting the lesion shape itself).
                p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
                border_mode_seg='constant',
                padding_value_seg=-1,
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            # stock: multiplier_range=(0.75, 1.25), apply_probability=0.15
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.6, 1.4)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.3
        ))
        transforms.append(RandomTransform(
            # stock: contrast_range=(0.75, 1.25), apply_probability=0.15
            ContrastTransform(
                contrast_range=BGContrast((0.6, 1.4)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.3
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            # stock: gamma=(0.7, 1.5), apply_probability=0.1
            GammaTransform(
                gamma=BGContrast((0.6, 1.6)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            # stock: gamma=(0.7, 1.5), apply_probability=0.3
            GammaTransform(
                gamma=BGContrast((0.6, 1.6)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.4
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=0.5
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=0.5
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


class nnUNetTrainerWideAugBaseline_500epochs(nnUNetTrainerWideAug):
    """Widened augmentation, Dice+CE loss (nnU-Net's stock loss), 500-epoch budget, dataset002.

    Built on the plain baseline rather than the best-performing loss/sampling
    condition -- see module docstring. ``baseline-500`` (already trained) vs.
    this trainer is the intended A/B.

    __init__ must declare nnU-Net's exact named parameters, not *args/**kwargs
    -- see the long comment on ``nnUNetTrainerLossVariants._Epochs250Mixin``
    for why (same bug class, avoided the same way).
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        import torch

        super().__init__(plans, configuration, fold, dataset_json,
                          device=device if device is not None else torch.device("cuda"))
        self.num_epochs = 500
        self.save_every = 10
