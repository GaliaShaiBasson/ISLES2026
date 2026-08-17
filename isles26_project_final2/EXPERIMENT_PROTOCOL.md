# Locked experimental protocol

Use this protocol before inspecting final test statistics. Record any deviation
in the report instead of silently changing the analysis after seeing results.

## Research question and primary comparison

Does fixed inverse-lesion-volume case sampling improve small-lesion
segmentation relative to standard nnU-Net without materially degrading medium-
or large-lesion performance? The primary comparison is `sampling` versus
`baseline`. Curriculum, Focal, and Tversky are secondary comparisons.

## Cohort and splitting

1. Convert the public ATLAS training data with `prepare` and record the number
   of discovered, excluded, and usable cases.
2. Run `preprocess`, then `make-splits --seed 2026` exactly once.
3. Archive `splits_final.json`. Sessions belonging to one subject must stay in
   the same fold.
4. Train every experiment on folds 0–4 with the same split file and nnU-Net
   configuration. Do not choose different folds for different methods.
5. Use only out-of-fold validation predictions for the primary comparison.

## Methods held constant

- Dataset, preprocessing plans, fold definitions, network configuration,
  augmentation, optimizer, schedule, number of epochs, checkpoint policy, and
  inference settings.
- Only the named loss or case-sampling rule changes between experiments.
- Record the exact command and generated run manifest for each fold.

## Outcomes

- Primary: case-level Dice and penalized HD95 in millimetres.
- Safety check: raw HD95, HD95-defined rate, and empty-prediction rate.
- Stratified: Dice/HD95 by lesion-size tertile and acquisition center.
- Exploratory: center × lesion-size cells and Dice versus continuous volume.

If exactly one mask is empty, raw HD95 is undefined and the penalized HD95 is
the physical image diagonal. Report the number of penalized cases for every
method. Never drop those cases silently.

## Statistical analysis

- Verify that every experiment contains exactly the same case IDs.
- Compare methods by paired case-level differences.
- Report mean and median improvement, 95% paired bootstrap confidence interval,
  Wilcoxon signed-rank p-value, and Holm-adjusted p-value.
- Positive improvement always means better: candidate minus baseline for Dice,
  baseline minus candidate for HD95.
- Treat sparse subgroup results as descriptive and show `n` for every cell.
- Do not claim equivalence from a non-significant p-value.

## Minimum evidence before conclusions

- All 25 scientific trainings completed (five methods × five folds), or a
  clearly disclosed reduced protocol approved by the instructor.
- No missing expected out-of-fold predictions.
- Training curves and manifests inspected for failures.
- `python isles26.py analyze` completed without the unpaired-cohort override.
- Representative successes and failures visually reviewed.

