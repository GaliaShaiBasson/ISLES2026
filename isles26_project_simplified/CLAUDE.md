# Project memory — read this first, every session

This file is the running record of **decisions and working agreements** for the
ISLES'26/ATLAS segmentation project. It exists so Claude doesn't re-litigate or
silently re-decide things already settled. Read it at the start of every
session; update it (see "Working agreement" below) whenever a new decision is
settled.

`PROJECT_PLAN.md` is the research execution plan (what to run, in what order).
This file is the *why* behind deviations from defaults, plus how we work
together. Cross-reference rather than duplicate.

## Working agreement

1. **Ask before assuming architectural placement.** Where a new script/module
   goes, how it wires into `isles26.py`, whether something becomes a new CLI
   subcommand vs. a standalone script — ask, don't guess silently, when more
   than one reasonable placement exists.
2. **Continuous knowledge updating.** When a new decision is settled in
   conversation, ask *"Should I update this file to reflect that?"* rather
   than either silently skipping documentation or silently rewriting project
   docs without flagging it.
3. **Docstring standard for new/edited scripts.** Every script's module-level
   docstring must cover, in this order:
   - **What it does**, one line.
   - **Expected input layout/format** (paths, file naming, CSV columns) if it
     reads external data.
   - **What it produces** (files written, side effects).
   - **Non-obvious rationale** — *why* a choice was made, when it isn't the
     "obvious" default (e.g. why a custom split instead of nnU-Net's default,
     why a field is parsed by name not position). If there's nothing
     non-obvious, this section can be omitted.
   - A `Usage:` example when the script is meant to be run standalone.
   See `data_prep/split_dataset.py` for the reference example.

## Decisions log

Newest first. Each entry: decision, rationale, where it's implemented.

### Split finalized (2026-08-17): 1,284/1,284 usable, OOD sites locked

- **OOD site list locked**: `R005, R008, R027, R029, R042, R070` (6 sites,
  n=129). Written into `data_prep/split_dataset.py` as `LOCKED_OOD_SITES`
  and used by default; pass `--ood-sites search` to fall back to the
  randomized search, or `--ood-sites A,B,C` to override explicitly. Locked
  rather than re-derived each run so later experiments are measured against
  the same held-out sites even if the script or search logic changes.
- **Final split sizes**: train=898 (48 sites), val=128 (37 sites),
  test_id=129 (37 sites), test_ood=129 (6 sites). Size-bin balance ~33/33/33
  large/medium/small in every split.
- **Empty-lesion cases (n=3 dataset-wide)**: all 3 landed in `train` (merged
  into the "small" bin for split mechanics only, since 3 is too few to
  survive a second stratified split as its own class). Left as-is rather
  than hand-moved into test — n=3 is too small to support a real
  empty-prediction conclusion either way.
- Output: `workspace/splits/{train,val,test_id,test_ood,manifest,broken_cases}.csv`.

### Custom stratified train/val/test_id/test_ood split (replaces nnU-Net's default 5-fold CV)

- **Decision:** Do not rely on nnU-Net's built-in `do_split()` (plain
  unstratified 5-fold `KFold`, seeded, over whatever is in `imagesTr`, no
  concept of held-out test data). Instead, generate our own split *before*
  `prepare` populates `imagesTr`.
- **Split structure:** `train` (70%) / `val` (10%) / `test_id` (10%) /
  `test_ood` (10%), single fixed split (not k-fold — see epoch-budget
  decision below for why).
  - `test_ood` = entire sites (5–8 of them) excluded from train/val/test_id
    completely. This is what actually measures cross-center generalization —
    the explicit reason ISLES'26 exists as ATLAS's successor (see
    `ISLES2026_challenge.md`). A model that only ever sees a site's
    scanner/protocol in training tells you nothing about generalization to
    an unseen site.
  - `test_id` = held-out subjects from sites also present in train, for a
    normal "same distribution" held-out number to report alongside OOD.
  - OOD site selection is a randomized search (seeded) over site subsets,
    scoring against both a target case-count (~10% of usable cases) and
    matching the overall lesion-size-bin proportions — not hand-picked, and
    deliberately excludes the largest few sites (removing one whole large
    site would blow past 10% and starve train diversity).
- **Stratification variable:** lesion-size bin (tertiles, via the existing
  `data_prep/metadata_utils.py:assign_size_bin` — reused rather than
  reinvented so bins match what `prepare`/sampling already use downstream).
  Zero-lesion ("empty") cases get their own bin category; kept mostly in
  train with 1–2 flowing into test naturally via their site's OOD/ID
  assignment, to support the empty-prediction/false-positive reporting
  `PROJECT_REVIEW.md` already calls for.
- **Implemented in:** `data_prep/split_dataset.py`. Finalized against the
  complete raw upload — see entry above for locked OOD sites and split sizes.
- **Still pending (not yet implemented):**
  1. Wire the split into `prepare_isles26_dataset.py` so only `train`+`val`
     case IDs are ever copied into `imagesTr` — `test_id`/`test_ood` must
     never enter nnU-Net's raw dataset at all, or nnU-Net's own fold logic
     could quietly train on them.
  2. Generate `splits_final.json` (single fold, `train`/`val` from our split)
     and place it in `nnUNet_preprocessed/DatasetXXX_.../` before training,
     so nnU-Net's `fold 0` uses our stratified split instead of generating
     its own unstratified random one.

### Raw-data integrity validation before use

- **Decision:** Every case's T1w + mask files are gzip-decodability-checked
  before being considered usable; broken cases are logged to
  `broken_cases.csv`, never silently dropped or silently included.
- **Rationale:** caught mid-upload corruption directly — many T1w scans were
  truncated to exactly 1,048,576 bytes (a transfer chunk boundary) while the
  upload was still in progress. This is a recurring risk (re-uploads,
  partial transfers), not a one-off fix for two files.
- **Implemented in:** `data_prep/split_dataset.py:discover_and_validate`.

### Metadata CSVs must be parsed by header name, never column position

- **Decision/finding:** the per-case `*_metadata.csv` files do **not** have a
  consistent column order across sites — all 169 `SOOP`-site files have
  `ATLAS2_DATASET,SESSION_ID,...` swapped relative to every other site's
  `SESSION_ID,ATLAS2_DATASET,...`. Positional parsing would silently produce
  wrong values for one whole site.
- Also noted: `CHRONICITY` is ~80% blank and single-valued where present;
  `ATLAS2_DATASET` is a legacy ATLAS2 tag, not an instruction for this
  project's split. Neither should be used as a stratification variable.

### Training epoch budget: 250 epochs (not nnU-Net's 1000 default), uniform across all real experiments

- **Decision:** `baseline`, `losses` (all 4 variants), and `sampling` all use
  a 250-epoch schedule instead of nnU-Net's default 1000.
- **Rationale:** ~4x GPU-time reduction (linear in epoch count, since
  iterations/epoch and batch size are unaffected). Applied *identically* to
  every condition so the baseline/loss/sampling comparison stays controlled
  — this is why the fix was "shrink the schedule uniformly," not
  patience-based early stopping (which would let different conditions train
  for different effective lengths depending on validation noise, a
  confound).
- **Implemented in:** `custom_trainers/nnUNetTrainerLossVariants.py`
  (`_Epochs250Mixin` + `*_250epochs` classes),
  `custom_trainers/nnUNetTrainerLesionAwareSampling.py`
  (`nnUNetTrainerLesionAwareSampling_250epochs`), wired into
  `TRAINER_GROUPS` in `isles26.py`. Baseline uses nnU-Net's own built-in
  `nnUNetTrainer_250epochs` (no custom class needed).
- Single fixed train/val split chosen over 5-fold CV for the same GPU-time
  reason — see split decision above.

### Data source: ATLAS R3.0 raw, native space only

- Documented in `ISLES2026_challenge.md` — cross-referenced here so it isn't
  missed. Must train/evaluate on raw (native-space, skull-stripped) data
  only, never the preprocessed/standardized archive, per the official
  ISLES'26 dataset page. Historical project scaffolding targeted ATLAS
  R2.1's folder layout; current downloads are R3.0, same layout shape.
