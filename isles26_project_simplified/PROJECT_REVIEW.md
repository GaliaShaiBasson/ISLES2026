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

#==========================================================#

## Checklist review: "General Training Tips" (medical imaging course deck)

Cross-checked the project against the course's 6-section practical checklist
(sanity checks/diagnosis, data representation, preprocessing discipline,
loss/optimization, checkpoints/ensembles, debugging workflow — section 7,
generative/fine-tuning, is not applicable here). Full deck:
`General Training Tips - Practical Checklist.pdf`.

### Already covered, in several places exceeding the checklist's bar

- **Diagnosis discipline.** `check_status.sh` surfaces live epoch/train-val
  loss/pseudo-Dice/EMA-Dice. The 2026-08-17 sampling-collapse bug was caught
  exactly the way the deck recommends: Dice pinned at 0.0 for 96 epochs while
  train/val loss kept moving — correctly read as an always-empty-prediction
  collapse, not slow learning (see CLAUDE.md decisions log).
- **Deliberate data representation.** Full 3D volumes via nnU-Net's
  `3d_fullres`, patch-based by construction — matches the deck's "small
  pathology → patches, high resolution" guidance for stroke lesions.
- **Preprocessing discipline.** Per-volume z-score normalization; gzip
  integrity validation caught real mid-upload corruption; metadata parsed by
  header name after finding a real site-level column-order bug (`SOOP` sites
  had swapped columns) — the checklist's "check preprocessing before
  blaming the model" already paid off once, concretely.
- **Patient/site-level splitting, beyond the checklist's ask.** Not just
  no-patient-leakage — whole *sites* are held out (`test_ood`) for genuine
  cross-center generalization on top of patient-level `test_id`. Val is used
  for selection, test for final evaluation, never tuned on.
- **Class imbalance handled explicitly and iterated on.** Lesion-size-bin
  stratified sampling (`nnUNetTrainerLesionAwareSampling*`), with a
  documented bug-fix history (continuous inverse-volume weighting →
  bin-level weighting after the collapse above).
- **Loss selection studied as a controlled experiment**, not a single
  choice: Dice, CE, Focal, Tversky, FocalTversky, DiceTversky all
  implemented and compared under identical epoch budgets.
- **Checkpointing plus provenance beyond the checklist's ask.** nnU-Net's
  own best/latest/periodic checkpoints, plus run-fingerprinting so
  hyperparameter changes can't silently collide with old results/checkpoints.
- **10-step "when a model performs badly" order** is effectively already
  operationalized by the fingerprinting/idempotency/integrity-check/
  auto-resume infrastructure in `isles26.py`.

### Gaps worth addressing

1. **No prediction ensembling.** The deck's "cheap trick" — averaging
   predictions across `epoch_80/90/100` checkpoints from one run, or across
   the already-trained loss-variant models — isn't implemented anywhere in
   `evaluation/`. Free accuracy given the checkpoints already on disk from 6+
   trained conditions.
2. **LR/weight decay/optimizer are untouched nnU-Net defaults** (SGD +
   PolyLR). Never revisited as a lever, even though patch/batch size was
   investigated (2026-08-18 GPU-usage entry). Not wrong, just undecided —
   the deck's LR-symptom→fix and batch-size↔LR thumb rules haven't been
   applied deliberately.
3. **No formal "overfit a tiny subset" sanity test as a standing check.**
   `nnUNetTrainerDebugFast` is a fast smoke test (5 epochs, sample dataset),
   not literally "take 8–16 cases, confirm near-zero loss." Given how
   directly this would have caught the sampling-collapse bug earlier, worth
   adding as a first-line check before trusting any new trainer.
4. **Augmentation strategy is default, not deliberate.** CLAUDE.md already
   flags this: brightness/contrast/gamma ranges are nnU-Net's generic mild
   defaults, elastic deformation is off, chosen for no reason specific to
   this dataset's cross-scanner generalization goal. Logged as a *future
   consideration*, not yet decided — the deck's "is augmentation
   anatomically valid / too strong" question hasn't been asked here yet.
5. **No EMA/SWA weight averaging** — the deck's one sanctioned exception to
   "don't average weights." Low priority (nnU-Net already tracks a
   pseudo-Dice EMA for checkpoint selection) but a real omission if chasing
   additional Dice cheaply.
