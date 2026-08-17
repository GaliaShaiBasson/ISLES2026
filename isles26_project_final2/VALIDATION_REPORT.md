# Local validation report

Date: 2026-08-17

## Passed

- Python bytecode compilation for all project modules.
- Full lightweight test discovery: 21 tests found, 20 passed, 1 skipped.
- Strict cohort rejection and paired-statistics synthetic tests.
- Deterministic grouped-split invariant tests.
- Synthetic generation of all expected analysis figures.
- CLI parsing and print-only smoke checks for multi-fold training and
  out-of-fold evaluation.

## Skipped or unavailable in this environment

- The differentiable Focal/Tversky loss test was skipped because PyTorch was
  not installed in the validation runtime.
- nnU-Net trainer discovery, real NIfTI I/O, GPU training, checkpoint creation,
  and ATLAS evaluation could not be executed because the required heavy
  packages, dataset, and trained outputs were not present.

Before reporting results, install the pinned dependencies, run `doctor`, run the
debug trainer, and complete the locked five-fold protocol in
`EXPERIMENT_PROTOCOL.md`.
