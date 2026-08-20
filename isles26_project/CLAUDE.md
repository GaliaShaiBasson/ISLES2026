# Project memory — read this first, every session

This file is the running record of **decisions and working agreements** for the
ISLES'26/ATLAS segmentation project. It exists so Claude doesn't re-litigate or
silently re-decide things already settled. Read it at the start of every
session; update it (see "Working agreement" below) whenever a new decision is
settled.

`PROJECT_PLAN.md` is the research execution plan (what to run, in what order).
This file is the *why* behind deviations from defaults, plus how we work
together. Cross-reference rather than duplicate.

## Checking on a running unattended experiment

When asked for run status, use `scripts/check_status.sh` (run from the project
root; it resolves paths itself regardless of CWD) rather than manually
grepping the log. It auto-picks the newest `workspace/logs/full_run*.log`
and reports: which condition is done/in-progress/pending, current epoch +
latest train/val loss + pseudo dice + best EMA dice for the in-progress
condition, an ETA to finish training, whether the driver script and
`nnUNetv2_train` process are actually alive, and any errors near the log tail.

```
scripts/check_status.sh                 # most recent full_run*.log
scripts/check_status.sh <path-to-log>    # a specific run's log
```

## Standing constraint: nnU-Net version is pinned, not just a dependency choice

Custom trainers rely on nnU-Net 2.8.1 internals (constructor signatures, deep
supervision weighting details — see "Foundational hardening pass" below).
`nnunetv2==2.8.1` is pinned deliberately. Upgrading nnU-Net is a code change
to this project, not a routine dependency bump — revalidate every custom
trainer against the new internals before trusting results from it.

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

### Primary metric switched to Dice, project-wide (2026-08-20)

- **Decision:** every script's `--primary-metric` default flips from `hd95_mm`
  to `dice` (tiebreak flips the other way, to `hd95_mm`) --
  `select_finalist_from_val.py`, `plot_finalist_selection.py`,
  `pca_model_redundancy.py`, `pca_probability_redundancy.py`,
  `ensembling/ensemble_val.py`, `ensembling/ensemble_test.py`, and
  `evaluation/summarize_postprocess_grid.py`'s `--selection-metric`/
  `--tiebreak-metric` (`dice_mean`/`hd95_mean`). `rank_trainers()`'s tiebreak
  logic already swapped bidirectionally before this change (`tiebreak_metric
  = "dice" if primary_metric != "dice" else "hd95_mm"`), so only each
  script's CLI *default* needed changing, not the ranking logic itself.
- **Why:** Dice is always defined (HD95 is NaN whenever prediction or ground
  truth is empty -- tracked via the existing `hd95_n_nan` column -- this
  project has real empty-lesion cases), lower-variance across a val/test set
  this size (119-129 cases per split) since it isn't dominated by a few
  outlier boundary-distance cases the way HD95 can be, and it's the metric
  virtually every comparable segmentation report/challenge leads with --
  more legible for the report's Experiments section. HD95 and lesion-F1
  remain fully reported everywhere, just no longer the tie-breaking/ranking
  axis by default.
- **Real, not just cosmetic, consequence: `postprocess_predictions.py`'s
  connected-component grid search needs re-running under
  `--selection-metric dice_mean`.** Its original "no-op wins" conclusion
  (see "Backfill complete"/two-night-plan entries below) was reached under
  hd95_mean-primary selection specifically because HD95 is the metric small
  spurious components distort most -- that was a deliberate, real reason to
  use HD95 for *that* search even though the general primary metric has now
  moved to Dice. The original run's numbers are reassuring (dice_mean moved
  by ≤0.0004 across all 20 combos), suggesting the conclusion likely still
  holds under dice-primary selection too -- but this needs confirming by an
  actual rerun, not assumed to carry over silently. **Rerun on the same val
  set as the original search** (`fold_0/validation/` predictions vs.
  `labelsTr`, 120 cases) -- never test_id/test_ood; that discipline (tune
  post-processing only on val, apply the one frozen choice to test exactly
  once) doesn't change just because the primary metric did. Not yet done as
  of this entry -- planned for 2026-08-21.

### Migration confirmed: isles26_project is now the active project, isles26_project_simplified to be archived (2026-08-20)

- **Decision:** the "duplicate first, verify, never touch the original" reorg
  below is done being cautious -- `isles26_project/` is now where all new
  work happens (code edits, new analysis scripts, ensembling work). `.env`
  paths, all real trained checkpoints (`nnUNet_results/` etc.), and the
  shared `.gitignore` entries were already reorganized/verified in the
  entry below; nothing new was needed to make the switch.
- **`isles26_project_simplified/` is not maintained going forward** -- it
  will likely move to `archive/` once nothing else still reads from it.
  Don't port new changes there; if something is found broken only in that
  copy, it's no longer worth fixing.

### Repo structure reorg: isles26_project_simplified duplicated into this dir, split into stage CLIs, paths reorganized (2026-08-20)

- **Context:** the project had accumulated real structural mess in
  `isles26_project_simplified/` -- duplicate `figures/`/`evaluation/`
  directories at two levels (code dir vs. output dir with the same name),
  loose scripts/logs at the project root instead of under `workspace/`,
  five separate "old stuff" archive locations, and a single ~1,300-line
  `isles26.py` mixing shared env/fingerprinting infrastructure with five
  distinct command groups. See `REPO_STRUCTURE_PLAN.md` for the full
  before/after audit.
- **Approach: duplicate first, reorganize the copy, never touch the
  original until everything is verified.** `isles26_project_simplified/`
  was copied whole to this directory (`isles26_project/`) and every change
  below was made only here -- the original stayed live/tracked/untouched
  throughout, so nothing was at risk if the reorg went wrong partway.
- **Directory reorg:** `docs/` (+ `docs/reference/` for the PDFs),
  `scripts/` (core pipeline drivers only -- `run_full_experiment.sh`,
  `run_500ep_full_experiment.sh`, `run_postprocess_grid.sh`,
  `check_status.sh`, `sanity_overfit_check.sh`), `workspace/evaluation/` ->
  `workspace/results/` (stopped colliding conceptually with the
  `evaluation/` code dir), `workspace/case_metadata_*.csv` ->
  `workspace/case_metadata/`, loose `workspace/*.log` -> `workspace/logs/`,
  prediction/postprocess output dirs -> `workspace/predictions/`,
  `workspace/figures/` split by topic (`learning_curves/`,
  `results_comparison/`, `threshold_analysis/`, alongside the pre-existing
  `augmentation_examples/`/`finalist_selection/`), all queue/launcher
  scripts (regardless of origin dir) consolidated into
  `workspace/queue_scripts/`, `analysis/finalist_selection/` grouped (the
  3 finalist-selection-specific analysis scripts, kept inside `analysis/`
  rather than merged into `ensembling/` -- a real distinction: `analysis/`
  never touches GPU/CPU, `ensembling/` actually runs predictions -- see
  `ensemble_val.py`'s own docstring). `workspace/splits/` and
  `workspace/splits_full/` renamed to `workspace/splits_dataset001/` and
  `workspace/splits_dataset002/` (dataset-id-based, self-documenting --
  "full" previously meant *fewer* cases than plain `splits/`, which read
  backwards). `training/`'s old per-condition `.sh`/`.ps1` launcher pairs
  (superseded by `run_full_experiment.sh` and predating `isles26.py`
  becoming cross-platform) archived to
  `workspace/archive/training_legacy_scripts/`.
- **`isles26.py` split into stage CLIs**, since it was one file mixing
  shared infra with five distinct command groups: `core.py` (env/.env
  loading, `run_command`, run-identity fingerprinting, checkpoint path
  resolution -- imported by every stage, so none of them duplicate this
  logic), `setup_cli.py` (init, doctor), `data_prep_cli.py` (prepare,
  preprocess), `train_cli.py` (train -- `TRAINER_GROUPS` lives here),
  `evaluate_cli.py` (evaluate, aggregate, plot). Each stage script is
  independently runnable on its own (e.g. `python train_cli.py
  baseline-500 --dataset-id 2`, no "train" subcommand needed) as well as
  via `isles26.py`, which is now a thin umbrella that imports each stage's
  `register()` function. `data_prep_cli.py`'s and `evaluate_cli.py`'s
  splits-dir/manifest auto-detection became genuinely dataset-id-aware in
  the process (previously both silently defaulted to the same
  `workspace/splits` regardless of `--dataset-id`, a gap the old single
  name papered over).
- **Follow-up (same day): `core.py` itself split into `core.py` +
  `run_identity.py`.** `core.py` initially held both generic env/subprocess
  infra *and* the run-identity fingerprinting system together, just because
  both were "shared" -- but checking actual imports showed
  `setup_cli.py`/`data_prep_cli.py` only ever use the generic half, never
  fingerprinting (`compute_run_fingerprint`, `run_id_from_fingerprint`,
  `_guard_run_fingerprint`, `SAMPLING_METADATA_BY_TRAINER`,
  `_output_folder`, `_find_latest_checkpoint`/`_find_existing_checkpoint`),
  which only `train_cli.py`/`evaluate_cli.py` need. Moved the fingerprinting
  cluster to a new `run_identity.py`, imported only by those two. Makes the
  usage boundary structural (those two stages have no import path to it)
  rather than a comment inside a shared file that's easy to silently cross
  later. Verified: reran the exact resenc-M+sampling `train`/`evaluate`
  commands (both via `isles26.py` and standalone `train_cli.py`/
  `evaluate_cli.py`) and confirmed byte-identical output before/after.
- **Real bugs found and fixed while verifying the reorg, not just
  mechanical renames:**
  - `ensembling/ensemble_val.py` did `sys.path.insert(...,
    ".../analysis")` then imported `select_finalist_from_val`/
    `plot_finalist_selection` as top-level modules -- broken by the
    `analysis/finalist_selection/` move until the insert was pointed one
    level deeper.
  - 11 files (`scripts/*.sh`, `workspace/queue_scripts/*.sh`,
    `ensembling/export_val_probabilities.sh`) hardcoded `cd`/
    `nnUNet_extTrainer` to the literal path
    `/home/galia/ISLES2026/isles26_project_simplified` -- would have
    silently run against the *old* project directory entirely from inside
    the new one. Fixed to point at `isles26_project`.
  - `scripts/check_status.sh` did `cd "$(dirname "$0")"`, which correctly
    landed at the project root when the script lived there -- but after
    moving into `scripts/`, that same line landed it *inside* `scripts/`,
    breaking its relative `workspace/...` log glob. Fixed to `cd
    "$(dirname "$0")/.."`.
- Verified live throughout, not just read over: `doctor` (incl. custom
  trainer discovery), `train --print-only`, `aggregate` (correctly found
  all 7 real run directories under the new `workspace/results/runs/`),
  `plot`, `prepare --print-only` for both dataset 1 and 2 (confirmed the
  new dataset-id-aware splits-dir resolution), every stage script's
  `--help` standalone and via the umbrella, and `bash -n` across every
  touched shell script.
- **Not done:** the dated entries below this one still describe file
  layouts as they were *at the time* (e.g. `workspace/evaluation/...`) --
  left as historical record, not retroactively rewritten. Only this file's
  evergreen instructions (like the `check_status.sh` line above) and
  `README.md`/`PROJECT_PLAN.md`'s undated operational sections were
  corrected to the new paths.

### DC+TopK10 dropped: same empty-prediction collapse as pure TopK10, on the overfit-check gate this time (2026-08-19 night / 2026-08-20)

- **Context:** `nnUNetTrainerDCTopk10` (`custom_trainers/nnUNetTrainerLossVariants.py`) was
  written as a hedge after `nnUNetTrainerTopk10_500epochs` (nnU-Net's stock pure-TopK loss,
  no Dice/CE) collapsed to an all-empty prediction on the real 500-epoch run -- the idea
  being that Dice's overlap term (empty prediction -> Dice=0, a strong penalty) would anchor
  against the same collapse while keeping TopK's hard-voxel focus.
- **Result:** `queue_dctopk10.sh`'s overfit-sanity-gate (100 epochs / 25 iters on the 6-case
  `Dataset999_ATLASsample`, the same gate that would have caught the 2026-08-17 sampling
  collapse in minutes instead of 96 epochs) caught this one *before* any real GPU time was
  spent: Pseudo Dice stayed at exactly 0.0 for all 100 epochs, never moving once, while
  train_loss did keep decreasing (0.93 -> ~0.30, noisy) -- the network is training, it just
  never produces any foreground overlap. Full log:
  `workspace/archive/logs/queue_dctopk10_20260819_175343.log`. The real 500-epoch run was
  correctly never launched.
- **Ruled out a wiring bug before dropping it:** `DC_and_topk_loss`'s construction
  (`batch_dice`/`smooth`/`do_bg=False`/`ddp` kwargs, `ignore_label` passthrough, `k=10`)
  matches nnU-Net's own stock `_build_loss` pattern exactly -- nothing project-specific to
  miswire, unlike the 2026-08-17 bug (patched-the-wrong-object) or the 2026-08-17 sampling
  bug (data-dependent weight collapse). Confirmed live: ran `DC_and_topk_loss` standalone on
  a synthetic imbalanced batch (small foreground cube in a mostly-background volume) --
  produced a finite loss and a finite, nonzero gradient. So this is a real optimization-
  dynamics failure on this dataset's extreme foreground imbalance (lesions often <1% of
  volume), not a bug to fix -- TopK's hardest-10%-of-voxels selection is apparently still
  background-dominated enough, even with Dice's penalty present, to let this dataset's
  optimizer settle on empty predictions. Same qualitative failure as pure TopK10, just one
  gate-catch cheaper.
- **Decision: dropped, not retried.** Two independent TopK-family losses collapsing the same
  way is itself the reportable finding -- log it as a real negative result (TopK-based losses
  are not robust to this dataset's imbalance, at least not without further intervention) next
  to the other loss-variant results, rather than spending more GPU/debugging time chasing a
  fix. 7 solid conditions (baseline, focal-tversky, tversky-mild, sampling, sampling-pow,
  wideaug, resenc-M architecture in progress) already cover the report's comparison needs.

### Curriculum (annealed-p) power-law sampling trainer added, not launched (2026-08-19)

- **Idea:** instead of a single fixed p in `sampling_weights_from_size_bin_power`
  (as `sampling-pow-500` uses, p=0.5 fixed for all 500 epochs), anneal p from 0
  (uniform/natural distribution) to 1 (full per-bin volume-mass correction) over
  the course of training. Same idea as "deferred re-weighting" in the long-tailed-
  recognition literature (e.g. Cao et al. 2019 LDAM-DRW) — let the network learn
  generic features on the natural distribution first, then push toward the rare
  small-lesion bin later, rather than reweighting from epoch 0 the way every other
  sampling trainer in this project does.
- **Real engineering constraint found while implementing:** nnU-Net's train
  dataloader runs inside worker processes spawned by `NonDetMultiThreadedAugmenter`
  (see `nnUNetTrainer.get_dataloaders`) — pickled copies, not shared with the main
  process. Mutating `sampling_probabilities` on the in-process loader object (the
  naive approach) would never reach the workers actually drawing batches. Two real
  options: (a) periodically tear down and rebuild the whole dataloader (fresh
  worker processes) with newly-computed weights, making p a step function; or
  (b) force a single-process dataloader (`num_processes=0`) and mutate weights
  continuously, at the cost of serializing data loading (likely a real slowdown,
  working against this project's GPU-time discipline). Chose (a): rebuild every 25
  epochs (20 steps across the 500-epoch budget), close enough to continuous
  without the (b) slowdown risk.
- **Implemented:** `nnUNetTrainerLesionAwareSamplingPowCurriculum` (base class,
  `custom_trainers/nnUNetTrainerLesionAwareSampling.py`) + `..._500epochs_full`
  (500-epoch mixin, `custom_trainers/nnUNetTrainer500epochs.py`), wired into
  `isles26.py` as `sampling-pow-curriculum-500`, an **additional** condition —
  `sampling-500` (4:2:1) and `sampling-pow-500` (fixed p=0.5) are untouched.
  Reads `workspace/case_metadata_full.csv` directly (size_bin +
  lesion_volume_mm3, same file `sampling-500` reads) rather than a precomputed
  `sampling_weight` column, since the weight formula must be re-evaluated at a
  new p on every rebuild — own dedicated env var
  (`ISLES26_SAMPLING_POW_CURRICULUM_METADATA_CSV_FULL`) so an override never
  leaks between trainers. `p` schedule constants (`P_START`/`P_END`/
  `DATALOADER_REBUILD_EVERY_EPOCHS`) live as class attributes, not config —
  changing the schedule needs a new trainer class/name, consistent with how
  every other hyperparameter in this project's trainers works.
- **Verified but NOT run:** both files compile, `python isles26.py train
  sampling-pow-curriculum-500 --dataset-id 2 --print-only` resolves the command
  correctly, and the step-schedule math was checked standalone (p pinned at
  exactly 0.0 through epoch 24, climbing every 25 epochs, exactly 1.0 from epoch
  475 through 499). Not launched — no GPU time left for a new 500-epoch
  condition before the report is due (see PROJECT_PLAN.md "Future work").

### Foundational hardening pass (baseline, predates the dated entries below)

A review of the original operational layer found it required several
environment variables, duplicated Bash/PowerShell scripts, manual custom-
trainer installation into `site-packages`, and hand-built evaluation
commands. Fixed as a batch, before per-decision dated entries started:

- **Debug trainer was incompatible with current nnU-Net.** Its constructor
  passed `unpack_dataset`, not accepted by nnU-Net 2.8.1 — fixed to match
  the current constructor signature.
- **Lesion-aware sampling patched the wrong object.** The parent method
  returns an augmenter after worker startup, not the raw loader that owns
  case indices — fixed to inject `sampling_probabilities` while the
  training loader is constructed (distinct from the later data-dependent
  weight-collapse bug below — this was a wiring bug, not a weighting-
  formula bug).
- **Custom trainers were installed into `site-packages`** (brittle across
  reinstalls/venvs) — switched to `nnUNet_extTrainer` so project code stays
  in the project.
- **Project identity was ambiguous**: named ISLES'26 but the converter and
  examples targeted ATLAS R2.1. Startup flow and converter docs now state
  the actual data source explicitly (see "Data source: ATLAS R3.0" below).
- **Data-prep file had duplicated module headers/imports** — reduced to one
  implementation with explicit path validation, duplicate case-ID checks,
  and safe overwrite behavior.
- **Correctness/robustness fixes bundled into the same pass:** pinned
  `nnunetv2==2.8.1` (was "any version newer than 2.4"); mirrored nnU-Net
  2.8.1's deep-supervision weighting incl. its DDP workaround; fixed
  ignore-label handling so positive ignore labels can't cause one-hot
  indexing failures; made focal-loss alpha class-specific for the binary
  case instead of a constant multiplier; made lesion-size binning work with
  small datasets and repeated lesion volumes; added prediction/ground-truth
  shape+affine checks before metric computation (this is the check that
  later caught the real `ATLAS_r028s017_ses1` affine mismatch, see
  "Full pipeline smoke test" entry below); made evaluation fail clearly on
  zero cases evaluated; added duplicate-row checks during aggregation;
  excluded non-positive lesion volumes from log-scale plots; removed unused
  direct dependencies.

### Optimizing GPU usage: investigated, no free lever found -- leave as-is (2026-08-18)

- **Context:** GPU is an L40S, 46 GB VRAM. Live during the 500-epoch/dataset002 run:
  99% compute utilization but only ~8.6 GB (~19%) VRAM used. Looked like real
  headroom being left on the table. Two concrete ways to use it were tested --
  both came back net negative, not free.
- **Bigger patch/batch on the same architecture (`-gpu_memory_target`):**
  previewed via `nnUNetv2_plan_experiment -d 2 -gpu_memory_target 32
  -overwrite_plans_name nnUNetPlansPreview32GB` (isolated -- never touched the
  live `nnUNetPlans.json` the running job was reading). Result: patch grows
  128x128x128 -> 160x192x160 (2.34x) and batch grows 2 -> 4 (2x) = **~4.69x
  more voxels processed per iteration**. GPU compute is already the bottleneck
  (99% util at the current small config), so this is expected to cost close to
  a proportional ~4.7x in wall-clock, turning tonight's ~18h estimate for 4
  conditions into ~80+ hours. Same order of magnitude as switching to
  ResEncL/XL (see the existing entry below) -- there's no cheap version of
  "use more memory," because the extra memory only exists to hold more compute.
- **Running two conditions concurrently on the same GPU:** a quick 5-epoch/
  20-iteration 2D debug-trainer test alongside `baseline-500` initially showed
  *no* measurable slowdown during its training phase, suggesting real idle
  compute headroom -- but that job was too small/short to be trustworthy.
  Re-tested at real scale: ran `baseline-500` and `focal-tversky-500`
  concurrently for ~12 minutes (both real 3d_fullres jobs). Result:
  `baseline-500` went from 30.3s/epoch solo to 63-65s/epoch concurrent (~2.1x
  slower); `focal-tversky-500` ran at 71-80s/epoch concurrent (vs. ~32s/epoch
  solo from the earlier 250-epoch run, ~2.3x slower). Net: finishing both
  conditions sequentially (~8.6h) is *faster* than running them concurrently
  (~10.4h, bounded by the slower of the two) -- concurrency is a genuine net
  loss here, not a free lunch, because the GPU's compute is actually saturated
  by one real job even though VRAM isn't.
- **Conclusion:** the unused VRAM is real but not spendable for free by any
  method tried. Don't revisit this speculatively -- only reopen it if
  architecture change (ResEnc, see below) becomes the actual next research
  question, at which point the cost is a real tradeoff to weigh, not a bug to
  fix.

### Backfill complete: full val/test_id/test_ood coverage for all 6 conditions (2026-08-17)

- Ran the deferred baseline/dice/focal held-out predict+evaluate once the
  main pipeline finished and the GPU was fully idle (no contention this
  time, unlike the earlier attempt). All three succeeded cleanly. Re-ran
  `aggregate`/`plot` to fold them in.
- **Consistent finding across all 6 conditions**: `test_ood` Dice is equal
  to or higher than `test_id` Dice in every single condition (baseline
  0.625->0.665, dice 0.602->0.638, focal 0.440->0.495, tversky 0.591->0.619,
  focal-tversky 0.593->0.623, sampling 0.608->0.661) -- no visible
  generalization gap in this split, which is the opposite of what ISLES'26's
  motivating premise (ATLAS'22 generalization gap) would predict. Worth
  investigating per-site (not just pooled OOD) before writing this up --
  could be genuine, could be an artifact of which 6 sites ended up in the
  locked OOD set.
- **sampling (4:2:1) vs baseline, by size bin (val)**: small 0.500 vs 0.494
  (+0.006, noise-level), medium 0.523 vs 0.586 (-0.062), large 0.798 vs
  0.819 (-0.021) -- real cost, marginal/unclear benefit. Motivated the
  `sampling-pow` (p=0.5) follow-up as the next informative data point
  (stronger correction, to distinguish "sampling doesn't help here" from
  "4:2:1 was too weak to show it") -- not run yet, GPU was occupied with
  the backfill; ready via `python isles26.py train sampling-pow --dataset-id 1`.

### Overnight run complete (2026-08-17), one post-hoc aggregate fix

- All 6 conditions attempted: baseline, dice, focal, tversky, focal-tversky,
  sampling. Pipeline exited cleanly ~18:07.
- **Final `aggregate` step failed once** on a duplicate-row conflict against
  `workspace/evaluation/results_val_only.csv` -- a manually-created combined
  file (from testing the `lesion_f1` addition mid-run) that got swept up by
  `aggregate`'s `results_*.csv` auto-glob alongside the real per-condition
  files, since it already contained `baseline_val`/`dice_val`/etc. rows. Not
  a bug in the duplicate check -- it correctly caught a real overlap. Fixed
  by moving the manual file (and its summary siblings) into
  `workspace/evaluation/manual_val_only_run/`, then re-ran `aggregate`/`plot`
  successfully. **Takeaway:** any manual/ad-hoc `--out-combined`/evaluate
  output placed directly in `workspace/evaluation/` risks colliding with the
  auto-glob the next time `aggregate` runs with no explicit file list --
  keep one-off combined files in a subdirectory instead.
- `compute_metrics.py`/`aggregate_results.py`/`plot_results.py` gained a
  `lesion_f1` metric (lesion-wise F1, with `--lesion-connectivity`/
  `--min-lesion-voxels` args) and `isles26.py` was updated to pass them
  through -- done by the user mid-run, already reflected in current results.

### Run-identity fingerprinting implemented -- results/checkpoints can no longer silently collide across hyperparameter changes (2026-08-17)

Closes the gotcha below. Chosen approach (of 3 options presented): **auto-fingerprint
from actual config**, no manually-typed run tag to remember.

- **`compute_run_fingerprint(trainer, dataset_id, dataset_name, configuration, fold, env)`**
  (`isles26.py`) captures everything that determines a run's results but that nnU-Net's
  own output-folder naming (trainer/plans/config/fold) can't see on its own: `num_epochs`
  and `save_every` (regex over `inspect.getsource(cls.__init__)`, run in a subprocess with
  `nnUNet_extTrainer` set -- doesn't rely on the trainer being named `..._250epochs`, so
  it also catches someone changing epochs without renaming the class), `n_folds_in_split`
  (length of the active `splits_final.json` in `nnUNet_preprocessed`, so a switch from our
  1-fold split to a real 5-fold CV doesn't silently reuse fold 0's folder), and, for
  lesion-aware-sampling trainers, a hash of the actual `(case_id, sampling_weight)` pairs
  in whichever metadata CSV that trainer reads (ties identity to weight *content*, so
  regenerating `case_metadata_pow_p05.csv` with a different `--p` is correctly seen as a
  different run even though the filename didn't change).
- **`run_id_from_fingerprint()`**: `{trainer}__{config}__fold{fold}__fp{10-hex-digest}` --
  readable prefix, but the hash is what actually guarantees no collision.
- **Checkpoints**: `cmd_train` now calls `_guard_run_fingerprint(output_folder, fingerprint,
  overwrite)` before the existing checkpoint skip-guard, for every trainer in the run
  (writes `isles26_fingerprint.json` inside nnU-Net's own output folder). No stored
  fingerprint yet (first time, or a pre-fingerprint legacy folder like the ones from
  tonight's overnight run) -> adopts the current fingerprint, no error. Stored fingerprint
  matches -> normal resume/rerun, unchanged behavior. Stored fingerprint differs -> refuses
  with a field-by-field diff unless `--overwrite`, in which case the old folder is moved
  aside to `<folder>.archived_<timestamp>/` (never deleted) before a fresh one starts.
  Verified live: adopted tonight's real `nnUNetTrainerBaseline_250epochs` folder without
  touching it, confirmed idempotent re-run, confirmed a synthetic `num_epochs` change is
  correctly refused without `--overwrite`.
- **Results**: `evaluate --trainer <name> [--dataset-id/--configuration/--fold] --split
  <val|test|...>` (replaces manually typing `--experiment "${group}_val"` /
  `--out-csv ...`) writes to `workspace/evaluation/runs/<run_id>/results_<split>.csv` plus
  a `run_manifest.json` (fingerprint + git commit + checkpoint path/mtime + pred/gt dirs +
  timestamp) in the same folder -- full provenance for later comparison/post-processing
  without re-running anything. If the target CSV already exists: no `--overwrite` ->
  `[skip]` (idempotent, matches `train`'s pattern -- re-running the pipeline unchanged
  never duplicates rows); `--overwrite` -> the previous CSV is archived into
  `runs/<run_id>/archive/<timestamp>_results_<split>.csv`, not deleted, before recomputing.
  `--trainer` is optional -- omitting it keeps the old flat `--experiment`-named layout for
  one-off/manual evaluate calls that aren't part of the tracked grid.
- **`aggregate`** now globs both the flat legacy layout and `runs/*/results_*.csv` (old
  results from before this change keep working unmodified), and additionally writes
  `runs_index.csv` -- one row per run, all manifest fields flattened -- a single flat
  catalog to compare/audit runs without opening JSON files.
- **`run_full_experiment.sh`** updated to call `evaluate --trainer "$trainer" --split
  val|test` instead of the old `--experiment "${group}_val"` form.
- Verified live end-to-end against the real completed `baseline` run: fingerprint computed
  correctly (`num_epochs=250, n_folds_in_split=1, sampling_weight_hash=None`), evaluate
  wrote real results to the new path, immediate re-run correctly skipped, `aggregate`
  picked up both layouts, `runs_index.csv` had the one real row -- then the test artifacts
  were removed and `aggregate`/`plot` re-run to restore the real 6-condition state exactly.

### Gotcha for later: a same-named variant run (e.g. a hypothetical 500-epoch version) would silently overwrite result CSVs, even though checkpoints are safe (2026-08-17) -- SUPERSEDED, see entry above

- **Checkpoints are safe automatically.** nnU-Net namespaces training output
  by exactly 4 things: trainer class name, plans identifier, configuration,
  fold (`nnUNet_results/Dataset{id}_{name}/{trainer}__{plans}__{config}/fold_{fold}/`).
  As long as a variant (different epoch count, different hyperparameter,
  whatever) gets its own trainer class name -- this project's existing
  convention, e.g. `nnUNetTrainerBaseline_250epochs` vs. a hypothetical
  `..._500epochs` -- checkpoints land in different folders automatically. No
  action needed there; this is already how every trainer in this project is
  named.
- **Result CSVs are NOT safe automatically -- real gap, not just a risk.**
  `run_full_experiment.sh` derives every evaluate/aggregate output name from
  the plain group name (`baseline`, `dice`, ...), never from the trainer
  class or epoch count: `--experiment "${group}_val"` ->
  `results_baseline_val.csv` regardless of which trainer variant produced it.
  A 500-epoch (or any other) variant run, reusing this script unmodified,
  would silently overwrite `results_baseline_val.csv`/`results_baseline_test.csv`
  and merge indistinguishably into `results.csv`/`summary_by_split.csv` --
  `aggregate`'s duplicate-row check wouldn't catch it either, since the
  experiment-name string would be identical, so it would look like a valid
  re-run rather than a collision.
- **Fix pattern for whenever this comes up** (not yet implemented -- deferred
  until an actual variant run is wanted): thread a variant tag through every
  artifact name for that run, either by suffixing every `--experiment`/
  `--out-csv` value (`"${group}_500ep_val"`) or, more simply, giving the
  whole variant run its own `--out-dir` (mirrors how the sample run got its
  own `workspace/sample_run/` tree, and `sampling-pow`'s dedicated metadata
  CSV/trainer name).
- **Do not edit `run_full_experiment.sh` in place while it's actively
  running** -- for a future variant run, copy it to a new script with the
  tag baked in rather than modifying the live file (risk of it re-reading a
  half-edited script mid-loop).

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

### test_id/test_ood now reachable for prediction + evaluation (2026-08-17)

- **Gap found:** `test_id`/`test_ood` cases were held out of `imagesTr`/
  `labelsTr` (correctly), but were never written anywhere else either — no
  ground truth to score against, no raw images in nnU-Net's flat naming for
  `nnUNetv2_predict` to run on. The whole point of building `test_ood` was
  to measure generalization, but there was no way to actually evaluate it.
- **Fix:** `prepare_isles26_dataset.py` now also writes `imagesTs`/`labelsTs`
  (nnU-Net's standard test-set layout) containing both `test_id` and
  `test_ood` cases combined — evaluate/aggregate distinguish ID vs OOD via
  the `split` column carried through `manifest.csv`, not via separate
  folders.
- `isles26.py evaluate` now defaults `--case-metadata-csv` to
  `workspace/splits/manifest.csv` (covers all 4 splits) instead of
  `workspace/case_metadata.csv` (train+val only, no `split` column) when the
  manifest exists.
- **Still pending:** `aggregate_results.py` only stratifies by `size_bin` —
  needs a `split`-grouped summary (ID vs OOD Dice) added the same way, and
  `evaluate`'s default `--gt-dir` (`labelsTr`) needs `labelsTs` passed
  explicitly when scoring held-out predictions.

### Sample/smoke-test pipeline runs must never touch real-run artifacts (2026-08-17)

- **Decision:** `--dataset-id 999` is reserved exclusively for sample/smoke-test
  runs (`split_dataset.py --sample-per-split`); the real dataset is always
  `--dataset-id 1`. This alone keeps `nnUNet_raw`/`nnUNet_preprocessed`/
  `nnUNet_results` fully separate, since nnU-Net namespaces everything under
  `Dataset{id:03d}_...`.
- All other sample-run outputs go under a dedicated `workspace/sample_run/`
  tree, via explicit CLI overrides (never the shared `.env` defaults):
  `--out-metadata-csv workspace/sample_run/case_metadata.csv`,
  `--case-metadata-csv workspace/splits_sample/manifest.csv`,
  `--out-csv .../workspace/sample_run/evaluation/results_*.csv`,
  `--out-dir workspace/sample_run/evaluation` (aggregate),
  `--out-dir workspace/sample_run/figures` (plot).
- **Rationale:** the real split (`workspace/splits/`), real prepared dataset
  (`Dataset001_ATLAS`), real `case_metadata.csv`, and real result CSVs must
  never be overwritten by a smoke test. `aggregate` didn't have an
  output-dir override before this (unlike `plot`, which already did) — added
  `--out-dir` to close that gap.
- `data_prep/split_dataset.py --sample-per-split N`: takes the *already
  final* real split and draws a stratified-by-size_bin subsample of ~N cases
  per split into a separate `--out-dir` (e.g. `workspace/splits_sample`) —
  does not re-derive OOD sites or split membership, just subsamples within
  the existing, reviewed split.

### Power-law sampling-ratio variant, prepared but not launched (2026-08-17)

- **Motivation:** the 4:2:1 fixed ratio used by `nnUNetTrainerLesionAwareSampling_250epochs`
  was an arbitrary geometric default, not derived from the data. Computed the
  real per-bin foreground-voxel mass on the actual training pool: despite
  equal case counts (342/342/342), **large lesions hold 91.5% of all
  foreground volume, small lesions only 0.9%** -- the true imbalance is far
  more extreme than case counts suggest. Full correction (equalizing total
  voxel exposure per bin) works out to ≈102:12:1, which would likely
  undertrain "large"; p=0.5 (sqrt-dampened) works out to ≈10:3.5:1, a
  principled middle ground grounded in the real numbers.
- **Implemented:** `metadata_utils.sampling_weights_from_size_bin_power(size_bins,
  volumes, p)` -- `weight(bin) ∝ (1/total_volume(bin))**p`; p=0 reproduces
  uniform/case-balanced, p=1 is full correction, p=0.5 is the recommended
  starting point.
  `data_prep/generate_pow_sampling_metadata.py` generates a standalone
  metadata CSV from this (never overwrites the original).
  `nnUNetTrainerLesionAwareSamplingPow_250epochs` (new trainer, `sampling-pow`
  group) reads it.
- **Isolation from the original sampling run, by design** (per explicit
  request): distinct trainer class name (own nnU-Net output folder
  automatically, since nnU-Net namespaces by trainer name) AND a distinct
  metadata CSV (`workspace/case_metadata_pow_p05.csv`, via a dedicated
  `CASE_METADATA_CSV_ENV_VAR`/`CASE_METADATA_CSV_DEFAULT` class-attribute
  pattern on `nnUNetTrainerLesionAwareSampling`, overridden per subclass).
  Also fixed `isles26.py`'s sampling-metadata pre-flight check, which
  previously only ever looked at the original CSV regardless of which
  sampling trainer was actually being launched.
- **Verified but NOT run**: command construction confirmed via
  `--print-only`, zero-volume-case weight share confirmed safe (0.6%, vs.
  the original bug's 66%), original in-progress `sampling` run confirmed
  undisturbed throughout. Deliberately not launched tonight -- would
  contend with the GPU for the currently-running condition, same mistake
  as the earlier backfill contention. Run later with
  `python isles26.py train sampling-pow --dataset-id 1`.

### Critical bug found and fixed mid-run: lesion-aware sampling weight collapse (2026-08-17)

- **Bug:** `sampling_weights_from_volume`'s continuous inverse-volume formula
  (`1/max(volume, floor=1.0mm3)`) is fragile against the real dataset's volume
  scale (median lesion ~4600mm3). On the real run, the **3 zero-volume
  (empty-mask) training cases alone captured 66% of all sampling
  probability**, and the top 10 cases (out of 898) captured 74.5% -- the
  network trained almost exclusively on a tiny near-empty subset and
  collapsed to a "predict nothing" degenerate solution. Caught by noticing
  Pseudo Dice pinned at exactly 0.0 for 96+ consecutive epochs (not slow
  learning -- checked train_loss/val_loss were both moving, only Dice was
  frozen at zero, which is the signature of an always-empty-prediction
  collapse) while every other condition tonight showed real, improving Dice
  from early epochs.
- **Why this wasn't caught earlier:** the pre-launch smoke tests (sample
  dataset, resume-logic verification) exercised `dice`/`nnUNetTrainer*`
  trainers, never actually ran `nnUNetTrainerLesionAwareSampling` itself
  through enough epochs to see its Pseudo Dice curve. Worth remembering for
  next time: a mixin/logic bug in a trainer's `__init__` will surface
  immediately on first instantiation (caught pre-launch, see below), but a
  *data-dependent* bug like this one only surfaces once real epochs run
  against the real data distribution -- smoke-testing on a 12-case sample
  with only 3 empty-lesion cases dataset-wide didn't reproduce the
  concentration effect at all.
- **Fix:** replaced continuous inverse-volume weighting with **bin-level**
  weighting (`sampling_weights_from_size_bin`, `metadata_utils.py`) -- every
  case in the existing `small`/`medium`/`large` tertile bin gets the same
  weight (4:2:1 ratio), so no single pathological case can dominate
  regardless of how close to zero its volume is. Verified: zero-volume
  cases' combined weight dropped from 66% to 0.5%, top-10 case concentration
  from 74.5% to 1.7%.
  `sampling_weights_from_volume` is kept in the file (unused) with a
  docstring warning, for reference/comparison only -- do not use it.
- **Recovery:** killed the in-progress (collapsed) training, moved its
  checkpoint aside to `workspace/broken_sampling_run_backup/` (not deleted --
  kept for reference), regenerated `workspace/case_metadata.csv`'s
  `sampling_weight` column in place with the fix (backup at
  `case_metadata.csv.bak-buggy-weights`). `run_full_experiment.sh`'s own
  retry loop picked the condition back up automatically with no checkpoint
  to resume from, so it started genuinely from epoch 0 with corrected
  weights -- confirmed this was necessary: resuming from the collapsed
  checkpoint would have continued training the already-broken model, not
  fixed it.
- **Verified fixed, not just theoretically**: watched the restarted run's
  Pseudo Dice live epoch-by-epoch (0, 0, 0, 0.004, 0.12, 0.38 across epochs
  0-5) -- climbing immediately, matching the healthy learning curve shape
  every other condition showed, versus the original run's 96+ consecutive
  epochs pinned at exactly 0.0.

### Critical bug found and fixed pre-launch: 250-epoch trainer __init__ signature (2026-08-17)

- **Bug:** `_Epochs250Mixin.__init__` (and `nnUNetTrainerLesionAwareSampling_250epochs.__init__`)
  used `def __init__(self, *args, **kwargs)`. nnU-Net's own `nnUNetTrainer.__init__`
  builds `self.my_init_kwargs` via `inspect.signature(self.__init__).parameters`
  -- since `self.__init__` resolves through the MRO to *our* mixin, that
  introspection collected `args`/`kwargs` as the parameter names instead of
  the real ones, and crashed with `KeyError: 'args'` the instant any
  250-epoch trainer was actually instantiated (not caught earlier because
  every previous smoke test only exercised `nnUNetTrainerDebugFast`, which
  never had this bug). This would have silently killed baseline, all 4 loss
  variants, and sampling on first real invocation of the overnight run.
- **Fix:** declare the exact same named parameters as
  `nnUNetTrainer.__init__` (`plans, configuration, fold, dataset_json,
  device`), matching the pattern `nnUNetTrainerDebugMixin` already used
  correctly. Caught and fixed by actually running a real (non-debug)
  250-epoch trainer on the sample dataset before trusting it for the
  overnight run -- this is why the "run the sample again" and "verify
  before trusting" steps mattered.
- Also added `nnUNetTrainerBaseline_250epochs` (plain `nnUNetTrainer` +
  `_Epochs250Mixin`) so baseline gets the same `save_every=10` treatment as
  every other condition -- previously baseline used nnunetv2's stock
  `nnUNetTrainer_250epochs`, which doesn't have it. `TRAINER_GROUPS["baseline"]`
  updated accordingly.

### Auto-resume on interrupted training (2026-08-17)

- **Decision:** `isles26.py train` now auto-detects a partial
  `checkpoint_latest.pth` (no `checkpoint_final.pth` yet) and automatically
  passes `--c` to resume, instead of nnU-Net's own default of silently
  restarting from epoch 0 and eventually overwriting the partial checkpoint.
  `--overwrite` opts back into a genuine restart; `--continue`/`--validate-only`
  bypass the check (already explicit about what they want).
- **Rationale:** this project now runs unattended for many hours
  (see `run_full_experiment.sh`) with nobody available to notice a crash and
  manually pass `--c`. A silent full restart on the next invocation would
  waste hours of GPU time without anyone knowing.
- Also dropped `save_every` from nnU-Net's default 50 epochs to 10 in every
  250-epoch trainer, shrinking the worst-case unsaved-progress window from
  ~40-60 min to ~7-12 min.
- **Verified live**, not just assumed: ran a real 3d_fullres trainer on the
  sample dataset to epoch 10, force-killed it, confirmed `checkpoint_latest.pth`
  existed and `checkpoint_final.pth` didn't, re-invoked the identical command,
  confirmed the `[resume]` log line and `--c` flag appeared, and confirmed the
  resumed run's new log picked up at exactly "Epoch 10" (not 0) with the
  correct continued learning-rate schedule.
- Added `--dataset-name` to `train` (previously only `prepare`/`preprocess`
  had it) -- needed to point `train` at the sample dataset
  (`Dataset999_ATLASsample`) for this verification; also just a real
  consistency gap on its own.
- **Implemented in:** `isles26.py:_find_latest_checkpoint`, wired into
  `_train_one`.

### Unattended overnight full-run script (2026-08-17)

- `run_full_experiment.sh`: preprocess dataset 1, then for each condition
  (baseline, dice, focal, tversky, focal-tversky, sampling, in that order)
  train -> predict on the held-out test set -> evaluate (val + held-out) ->
  incrementally re-aggregate + re-plot. No `set -e`; each condition's
  training gets up to 3 attempts (each re-invocation auto-resumes via the
  mechanism above) before the script gives up on that condition and moves
  to the next -- a crash never aborts the whole run or silently drops a
  condition without at least trying to resume it.
- Launched against the real dataset (id 1, `Dataset001_ATLAS`, 1,026
  train+val cases) after all of the above was verified on the sample
  dataset first.

### dice_by_split.png added (2026-08-17)

- `analysis/plot_results.py:save_by_split` — boxplot of Dice by
  train/val/test_id/test_ood x experiment, mirroring `save_by_size`'s
  layout. Gives the ID-vs-OOD generalization comparison a figure to go with
  `summary_by_split.csv`, which previously only existed as a table.
  Verified rendering correctly against the sample smoke-test results.

### Full pipeline smoke test passed end-to-end (2026-08-17)

- Ran split (6/split sample) -> prepare (imagesTr/labelsTr + imagesTs/labelsTs)
  -> preprocess -> `train debug` (GPU) -> `nnUNetv2_predict` on the held-out
  set -> evaluate (val, and test_id/test_ood separately) -> aggregate (incl.
  the new split-stratified summary) -> plot, entirely on `Dataset999_ATLASsample`
  / `workspace/sample_run/`. Real-run artifacts confirmed untouched throughout.
- **Bug found and fixed:** `cmd_aggregate` gained `--out-summary-by-split`
  support in `aggregate_results.py` but `isles26.py` never passed the flag
  through, so it silently fell back to a CWD-relative default and landed in
  the project root instead of the results dir. Fixed.
- **Real-data finding (not a pipeline bug):** one held-out case
  (`ATLAS_r028s017_ses1`) failed evaluation with a genuine affine mismatch —
  its ground-truth mask has a real ~4 degree rotation in its affine (nnU-Net's
  own `verify_dataset_integrity` already flagged an image/segmentation
  direction mismatch for this exact case during `preprocess`), while the
  exported prediction came back purely axis-aligned, ~22-32mm off in
  translation. `evaluate`'s existing affine check caught it correctly and
  errored instead of silently computing a wrong Dice. Expect a small number
  of similar cases at full scale (1,284 cases) — worth a pass counting how
  many, but not a blocker; the check is doing its job.
- `train debug` defaults to CPU by design (see README/PROJECT_PLAN) but ran
  noticeably slowly at batch_size=66 on CPU (~140s/epoch); `--device cuda`
  brought a 5-epoch run down to under a minute once past CUDA warmup. Both
  are fine for a smoke test since it's disposable either way; CPU stays the
  documented default (must work without a GPU), GPU is worth using when one
  is idle.

### Preprocessing auto-scales worker processes to the machine (2026-08-17)

- **Decision:** `preprocess` no longer uses nnU-Net's hardcoded defaults
  (`-np`/`-npfp` = 8) — it auto-detects usable CPU count via
  `os.sched_getaffinity(0)` (falls back to `os.cpu_count()`), reserves 2
  cores, and caps at 32 (preprocessing workers each hold a full volume in
  RAM; parallelism benefit plateaus well before core count does on a very
  large machine). Override with `--num-processes` or
  `ISLES26_PREPROCESS_NUM_PROCESSES` in `.env`.
- **Rationale:** preprocessing is GPU-free by design in nnU-Net (checked
  `DefaultPreprocessor` directly — no torch/cuda/device references
  anywhere in it; it's pure NumPy/SimpleITK resampling + I/O), so CPU
  parallelism is the only real speed lever for this step. Auto-detecting
  keeps this correct if the pipeline ever runs on a different machine,
  rather than hardcoding a number tied to this box's 144 cores.
- This is pure infrastructure — doesn't affect determinism or results, so
  (unlike the epoch/split decisions) no controlled-comparison concern here.
- **Implemented in:** `isles26.py:detect_num_processes`, wired into
  `cmd_preprocess`.

### Idempotency guards for preprocess/train (2026-08-17)

- **`preprocess`**: errors if `nnUNet_preprocessed/<dataset>/nnUNetPlans.json`
  already exists; `--overwrite` forces a redo (`--clean` passed through to
  nnU-Net). Matches `prepare`'s existing exists-unless-`--overwrite` behavior.
- **`train`**: nnU-Net itself has no skip-if-done behavior (only `--c` to
  resume) so this is our own guard, checking for
  `checkpoint_final.pth` under the standard
  `{trainer}__nnUNetPlans__{configuration}/fold_{fold}/` path.
  - Single explicit trainer (`--trainer X` or a single-trainer group like
    `dice`/`baseline`): **hard-aborts** if already trained — nothing else to
    fall through to.
  - Multi-trainer group (`losses`, 4 trainers): **skips** the already-done
    trainer and continues with the rest — hard-aborting the whole group on
    the first completed member would defeat resuming an interrupted study.
  - `--overwrite`, `--continue`, `--validate-only`, `--print-only` all bypass
    the guard (they're explicitly asking to touch the existing checkpoint).

### Split wired into prepare/preprocess (2026-08-17)

- `prepare_isles26_dataset.py --splits-dir`: only `train`+`val` case IDs are
  copied to `imagesTr`/`labelsTr`; `test_id`/`test_ood` are never written to
  the nnU-Net raw dataset. Also writes `splits_final.json` (single fold, our
  train/val ids) next to `dataset.json`.
- `isles26.py prepare` auto-passes `--splits-dir workspace/splits` when that
  directory exists (warns loudly and falls back to all-cases if missing);
  `--no-split` opts back into legacy behavior.
- `isles26.py preprocess` copies that `splits_final.json` into
  `nnUNet_preprocessed/<dataset>/` after `nnUNetv2_plan_and_preprocess` runs,
  so nnU-Net's `do_split()` picks it up instead of generating its own
  unstratified random fold.
- Verified live: 1,026 cases in `imagesTr` (898 train + 128 val), 0
  test_id/test_ood leakage, `splits_final.json` fold matches exactly.

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

## Future considerations (discussed, not decided/implemented)

Ideas raised in conversation that are explicitly *not* yet adopted — logged
so they aren't silently lost, and re-litigated from scratch next time
cross-center generalization comes up.

### Switch to nnU-Net's ResEnc planner (M or L) instead of the default planner (2026-08-17)

- **Context:** `nnUNetv2_plan_and_preprocess` (called with no `-pl` flag in
  `isles26.py:cmd_preprocess`) warns on every run that it's using nnU-Net's
  old default planner (`ExperimentPlanner`, plain U-Net encoder) and points
  at the newer residual-encoder presets, which nnU-Net's own benchmarks show
  outperforming the default at the same or modest extra GPU cost.
- **GPU available:** NVIDIA L40S, 46 GB VRAM (~37 GB free at last check) —
  comfortably covers `nnUNetPlannerResEncM` (similar budget to default) and
  likely `nnUNetPlannerResEncL` (~24 GB) too; `nnUNetPlannerResEncXL`
  (~40 GB) would be tight against other GPU usage on the box.
- **Not implemented / not decided which of M vs. L.** This is an
  architecture change, not a tuning knob — it would need to apply uniformly
  across baseline/losses/sampling to keep the comparison controlled (same
  principle as the 250-epoch decision above), and picking between M and L
  is a real tradeoff (L likely stronger but slower/more memory) worth a
  short discussion before committing, not a default swap-in. Revisit
  alongside or after the cross-center generalization augmentation work
  above.

### Cross-center generalization: augmentation tuning before anything adversarial (2026-08-17)

- **Question raised:** would a domain-adversarial training step (gradient-
  reversal domain classifier on encoder features, pushing toward
  site-invariant representations) help `test_ood` generalization?
- **Assessment:** plausible but not the first lever to pull — adversarial
  segmentation objectives are prone to training instability, especially at
  this dataset's size (hundreds, not tens of thousands, of cases), and risk
  suppressing real signal if lesion characteristics correlate with site
  prevalence. Cheaper levers likely to capture most of the benefit first:
  - **Widen intensity augmentation ranges.** Current ranges (stock nnU-Net,
    `nnUNetTrainer.get_training_transforms`, unmodified by any custom
    trainer here) are mild: brightness/contrast ×0.75–1.25, gamma 0.7–1.5,
    each applied at only 10–30% probability; elastic deformation is
    disabled entirely (`p_elastic_deform=0`). These weren't chosen for
    cross-scanner robustness — they're nnU-Net's generic defaults.
  - **Normalization is already per-case z-score** (`ZScoreNormalization`,
    mean/std from each volume's own foreground, no clipping, no shared
    reference stats) — gives free invariance to per-scanner gain/offset but
    nothing for non-linear differences (bias field, noise character,
    resolution).
  - **Report `test_ood` per-center, not just pooled.** Confirmed via
    `workspace/splits/*.csv`: 48 centers span train/val/test_id (very
    unbalanced — e.g. R009 contributes 83/12/16 cases across train/val/
    test_id, while R016/R020/R063 each contribute exactly 1 training case);
    `test_ood`'s 6 locked sites (R005, R008, R027, R029, R042, R070) are
    also unbalanced among themselves (R005/R042/R027: 32–37 cases each;
    R008: 7) — a pooled OOD Dice will be dominated by 3 of the 6 sites.
- **Not implemented.** If revisited: try widened brightness/contrast/gamma
  ranges via a new trainer subclass first (cheap, one afternoon), check
  whether the in-distribution vs. OOD gap shrinks, and only reach for a
  domain-adversarial head if a meaningful gap remains after that.

### Data source: ATLAS R3.0 raw, native space only

- Documented in `ISLES2026_challenge.md` — cross-referenced here so it isn't
  missed. Must train/evaluate on raw (native-space, skull-stripped) data
  only, never the preprocessed/standardized archive, per the official
  ISLES'26 dataset page. Historical project scaffolding targeted ATLAS
  R2.1's folder layout; current downloads are R3.0, same layout shape.
