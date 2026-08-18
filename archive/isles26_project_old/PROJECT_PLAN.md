# Project timeline

## Phase 0 — Setup (0.5 day)
- Install `nnunetv2`, set `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` env vars.
- Download ISLES'26, inspect 2-3 cases manually (spacing, orientation, whether
  masks are binary lesion masks or multi-class).
- Adjust `data_prep/prepare_isles26_dataset.py` to match the real folder layout.

## Phase 1 — Baseline (2-3 days)
- Run `data_prep/prepare_isles26_dataset.py` → nnU-Net raw dataset + `case_metadata.csv`.
- `nnUNetv2_plan_and_preprocess -d <ID> --verify_dataset_integrity`.
- `training/run_baseline.sh` — train default `nnUNetTrainer`, at least 1 fold
  (ideally the standard 5-fold CV if GPU time allows; if not, 1 fold + a held-out
  test split is an acceptable fallback for a course project — say so explicitly
  in the report).
- Run inference on validation/test cases, then `evaluation/compute_metrics.py`.
- **Deliverable:** baseline Dice/HD95 table + a couple of qualitative failure
  examples (smallest and largest lesions, best and worst Dice).

## Phase 2 — Loss function study (2-3 days)
- `custom_trainers/install_custom_trainers.py`.
- `training/run_loss_experiments.sh` trains: Dice-only, Dice+CE (nnU-Net
  default — you already have this from Phase 1), Focal, Tversky, Focal-Tversky.
- Same fold split as baseline for a fair comparison.
- `evaluation/compute_metrics.py` for each, `evaluation/aggregate_results.py`
  to combine into one table.
- **Deliverable:** bar chart of Dice/HD95 per loss, plus the size-stratified
  version (this is the plot that makes the "why" argument).

## Phase 3 — Lesion-aware sampling (1-2 days)
- Train `nnUNetTrainerLesionAwareSampling` (best loss from Phase 2, or Dice+CE
  if you want to isolate the sampling effect alone — pick one and say why).
- Compare against the equivalent run without lesion-aware sampling.
- **Deliverable:** does it help specifically on the small-lesion bin? Does it
  hurt large lesions (a classic trade-off)?

## Phase 4 — Analysis & writeup (2 days)
- Stratify all runs by: lesion-size bin, center (if metadata available),
  chronicity/days-post-stroke (if available).
- `analysis/plot_results.py` for final figures.
- Write up: baseline, method, results, failure cases, discussion of *why* the
  best method wins (tie back to the imbalance argument), limitations (single
  GPU, possibly reduced fold count, held-out split choices).

## Fallbacks if GPU time runs out
- Drop to 1 fold instead of 5, but keep the same fold for every experiment
  so comparisons stay fair.
- Reduce to 3 loss variants (Dice, Dice+CE, Focal-Tversky) instead of 5 if
  Phase 2 is squeezed.
- If lesion-aware sampling (Phase 3) doesn't fit, the loss study alone with
  solid stratified analysis is still a complete, defensible project.
