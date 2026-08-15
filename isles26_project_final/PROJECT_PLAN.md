# Project plan

## Research question

Does case-level lesion-volume-aware sampling improve small-lesion segmentation in nnU-Net without sacrificing performance on medium and large lesions, and are any gains consistent across acquisition centers?

## Experiments

1. Standard nnU-Net baseline.
2. Focal-loss nnU-Net.
3. Tversky-loss nnU-Net.
4. Fixed lesion-size-aware sampling: probability proportional to inverse lesion volume throughout training.
5. Curriculum lesion-size-aware sampling: uniform sampling in the first third, inverse-square-root volume in the middle third, inverse volume in the final third.

Size-balanced categorical sampling is intentionally excluded.

## Primary outcomes

- Dice
- HD95 (mm)

## Stratified analyses

- Overall performance.
- Performance by lesion-size tertile (`small`, `medium`, `large`).
- Performance by acquisition center.
- Performance by acquisition center × lesion-size tertile.
- Dice versus continuous lesion volume.

## Interpretation

The primary comparison is baseline versus fixed sampling. Curriculum sampling tests whether introducing the small-lesion bias progressively is preferable to applying the full bias from the start. Focal and Tversky provide loss-based imbalance controls.
