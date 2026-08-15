# Implementation notes

This version focuses the project on five experiments: baseline, Focal, Tversky, fixed lesion-volume-aware sampling, and curriculum lesion-volume-aware sampling. Size-balanced categorical sampling is not implemented.

Operational changes from the earlier revision:

- `init` updates supplied settings instead of silently preserving the old raw-root path.
- `.env` is project-authoritative, preventing stale inherited environment variables from overriding it.
- Metadata is stored per nnU-Net dataset ID/name instead of in one global CSV.
- `predict` is available through the unified runner.
- `evaluate` accepts dataset ID/name and attaches center metadata.
- `analyze` combines aggregation and plotting.
- Analysis now includes overall, lesion-size, center, and center×size summaries.
- PyTorch compatibility checking follows nnU-Net 2.8.1 package metadata rather than the earlier incorrect `<=2.8` rule.

The curriculum sampler changes a continuous inverse-volume exponent rather than using lesion-size categories. The two phase transitions rebuild the data augmenters so new sampling probabilities propagate to worker processes on both Windows and Linux.
