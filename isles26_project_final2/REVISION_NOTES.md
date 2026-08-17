# Revision notes

## Scientific-validity fixes

- Evaluation now uses an explicit expected cohort and fails on missing or extra
  predictions by default.
- Cross-experiment aggregation rejects unequal case sets and duplicate
  case/experiment rows.
- Empty-versus-nonempty masks retain raw undefined HD95 and receive an explicit
  physical-image-diagonal value in `hd95_penalized_mm`.
- Empty-prediction and HD95-defined rates are included in summaries and plots.
- Paired baseline comparisons now include bootstrap confidence intervals,
  Wilcoxon signed-rank tests, and Holm multiple-comparison correction.
- Center × size heatmaps display sample counts.

## Reproducibility fixes

- Added deterministic subject-grouped, center/size-aware five-fold split
  generation.
- Added multi-fold training and strict out-of-fold validation evaluation.
- Added per-training run manifests with commands, package versions,
  configuration, platform, and source hashes.
- Added sampling-distribution diagnostics, including effective sample size.
- Corrected the PyTorch minimum-version check for 2.1.2.

## Documentation and verification

- Added the locked experiment protocol and final-report checklist.
- Added ATLAS and nnU-Net citations and explicit no-results-yet language.
- Added full-cross-validation launch scripts for Bash and PowerShell.
- Expanded the lightweight suite to 21 tests; the PyTorch loss test skips only
  when PyTorch is not installed.

The revision does not include fabricated training results, checkpoints, or a
finished paper. Those require the licensed dataset, the configured compute
environment, author information, and completed experiments.
