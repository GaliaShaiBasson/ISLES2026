# Project review

## Overall assessment

The research structure is coherent: one baseline, a focused loss study, lesion-size-aware sampling, and stratified evaluation. The original operational layer was the weak point. It required several environment variables, duplicated Bash and PowerShell scripts, manual custom-trainer installation, and hand-built evaluation commands.

## High-impact findings

1. **The debug trainer was incompatible with current nnU-Net.** Its constructor included `unpack_dataset`, which is not accepted by nnU-Net 2.8.1. The revised trainer uses the current constructor signature.

2. **Lesion-aware sampling patched the wrong object.** The parent method returns an augmenter after worker startup, not the raw loader that owns case indices. The previous patch would either fail on missing attributes or fail to affect worker sampling. The revised trainer injects `sampling_probabilities` while the underlying training loader is constructed.

3. **Custom trainers were copied into `site-packages`.** This is brittle across reinstalls and virtual environments. The revised runner uses `nnUNet_extTrainer`, so project code remains in the project.

4. **The data identity was ambiguous.** The project is named ISLES'26, but the actual converter and examples target ATLAS R2.1. The converter documentation and startup flow now state this explicitly.

5. **The data-preparation file contained duplicated module headers and imports.** It has been reduced to one implementation with explicit path validation, duplicate case-ID checks, and safe overwrite behavior.

## Correctness and robustness changes

- Pinned `nnunetv2==2.8.1` instead of accepting any version newer than 2.4.
- Mirrored nnU-Net 2.8.1's deep-supervision weighting, including its DDP workaround.
- Corrected ignore-label handling so positive ignore labels cannot cause one-hot indexing failures.
- Made focal-loss alpha class-specific for the binary case instead of a constant multiplier.
- Made lesion-size binning work with small datasets and repeated lesion volumes.
- Added prediction/ground-truth shape and affine checks before metric computation.
- Made evaluation fail clearly when zero cases are evaluated.
- Added duplicate-row checks during aggregation.
- Excluded non-positive lesion volumes from log-scale plots.
- Removed unused direct dependencies.

## Remaining risks

- The ATLAS discovery patterns still need confirmation against the exact dataset download.
- The custom trainers are intentionally pinned to nnU-Net 2.8.1 internals. Upgrading nnU-Net should be treated as a code change and revalidated.
- Inverse lesion-volume sampling can heavily overweight the smallest cases. The resulting probability distribution should be inspected before interpreting the experiment.
- A single fold is useful for iteration but insufficient for strong claims. Keep folds and splits identical across methods.
