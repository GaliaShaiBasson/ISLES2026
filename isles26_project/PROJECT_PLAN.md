# Project plan and status

This file is *what's done, what's running, and what's left* before the
report is due. See `README.md` for setup and the exact CLI commands, and
`CLAUDE.md` for *why* each deviation from nnU-Net/project defaults was made
and for the full decisions log.

**This directory (`isles26_project/`) is now the active project (2026-08-20
onward)**, per the reorganization described in `CLAUDE.md`'s "Repo structure
reorg" entry. `isles26_project_simplified/` is being phased out and will
likely move to `archive/` once nothing else needs it -- new work happens
here, not there.

## Phase order and controlled-comparison rules

Split → prepare → preprocess → debug smoke test → baseline → loss study →
sampling → evaluate/aggregate/plot (see `README.md` for the commands). A few
rules apply across all of it, not obvious from the commands alone:

- Use the same fold, configuration, and epoch budget (250, later 500 — see
  `CLAUDE.md`) for every real comparison — changing more than one axis at a
  time makes the comparison uninterpretable.
- Don't proceed past the split step until `broken_cases.csv` is small/stable
  (i.e. the raw-data upload has actually finished).
- Before trusting a `sampling` run, inspect the relevant
  `workspace/case_metadata/case_metadata*.csv`'s `sampling_weight` distribution.
- Never treat the `debug` trainer's output as an experimental result — it's
  a pipeline check only.

## Status as of 2026-08-19 (post two-night plan)

The two-night plan below **completed**: all 7 conditions now have finished
500-epoch/`Dataset002` checkpoints, predictions, and evaluation —
`baseline`, `focal-tversky`, `tversky-mild`, `sampling` (4:2:1, full),
`sampling-pow` (p=0.5, full — the one item explicitly deferred two nights
ago, now closed), `wideaug`. `workspace/evaluation/runs_index.csv` has all 7
rows.

**Also landed since the two-night plan, not anticipated in it:**
- **Connected-component grid search finished and scored on held-out test**
  for `baseline-500` (`workspace/postprocess_grid/baseline500/`). Real
  finding, not just a completed task: **the auto-selected winner (lowest val
  HD95) is `min_voxels=1, connectivity=1` — the raw/no-op control itself.**
  Across all 20 combos, `dice_mean` moves by ≤0.0004 and `hd95_mean` only
  ever gets *worse* than the no-op as `min_voxels` grows (19.70mm at
  `min_voxels=1` → 20.3+mm at `min_voxels=20`); `lesion_f1` improves
  monotonically with more aggressive filtering, but never enough to move the
  HD95-based selection rule off the no-op. Contrast with the earlier
  hand-picked `min_voxels=10` test on `DiceOnly_250epochs`/`Dataset001`
  (`+0.035` lesion F1 but `+0.35mm` worse HD95) — same qualitative tradeoff,
  but the *properly selected* combo at 500 epochs concludes "don't
  post-process" rather than landing on a specific threshold. Frozen held-out
  test scoring (raw): 249/250 cases evaluable, Dice mean 0.653, HD95 mean
  17.94mm, lesion F1 mean 0.637 (`test_final/results_raw_test.csv`). Report
  this as a real negative result, not a gap — it's a legitimate
  Experiments-section row (see "Report deliverables" below).
- **Per-case softmax probabilities saved for all 6 non-baseline 500-epoch
  conditions** (`predict_prob_remaining.sh`, sequential — same
  GPU-contention reasoning as training) plus baseline separately
  (`predict_baseline500_prob_*.log`), each into its own `predTs_prob/`
  alongside the untouched `predTs/`. This is the prerequisite data for
  prediction ensembling (still deferred, see below) *and* unlocks two new
  exploratory scripts:
  - `analysis/dice_vs_threshold.py` — sweeps the foreground-probability
    threshold (nnU-Net's argmax export is an implicit fixed 0.5) and plots
    Dice vs. threshold per experiment, marking the current 0.5 operating
    point against the best-found one. Not yet run against the real 500-epoch
    probability exports — next exploratory step, cheap (CPU-only, no rerun
    of inference needed).
  - `analysis/probability_histogram.py` — per-experiment histogram of
    predicted foreground probability split by true GT class (lesion vs.
    background), log-scale y-axis (background voxels vastly outnumber
    lesion voxels). Same status: script exists and is committed, not yet run
    against the real data.
- **Training-curve parser built and run** (`analysis/plot_learning_curves.py`)
  — closes item 1 of the "needs a script" list below. Parses nnU-Net's
  `training_log_*.txt` (train/val loss, pseudo-Dice, LR, epoch time) into a
  CSV plus a 3-panel comparison figure across named runs, with a fixed
  regex bug avoided (scientific-notation LR values like `7e-05` don't get
  truncated to `7`); also merges multiple `training_log_*.txt` files per
  trainer (nnU-Net starts a new one on every resume/validate-only
  invocation) so a curve covers the full training history, not just the
  last invocation. Two run modes: explicit `--log 'label=path'` pairs
  (e.g. a 250-vs-500-epoch baseline comparison), or the default — auto-
  discover every trainer from `workspace/evaluation/runs_index.csv`
  (`isles26.py aggregate`'s output), narrowable with `--trainer`. Added a
  fourth output, `train_val_overlay_<tag>.png` — one small-multiple subplot
  per trainer with train loss and val loss overlaid on the same axes (the
  train/val gap reads directly off the two lines' vertical distance,
  instead of comparing two separate panels). **Run 2026-08-19** against the
  current 6 500-epoch trainers (auto-discovery mode, `--tag all_trainers`)
  → `figures/learning_curves_all_trainers.png`,
  `figures/lr_schedule_all_trainers.png`,
  `figures/train_val_overlay_all_trainers.png` — no condition shows a
  meaningful train/val divergence through epoch 500. The 250-vs-500(-vs-1000)
  baseline comparison is left as a documented `--log` example in the
  script's docstring, not run (the 1000-epoch condition doesn't exist yet).
- **`sampling-pow` at 250 epochs / `Dataset001` removed** (was leftover from
  before the 500-epoch/`Dataset002` port; the real 500-epoch version now
  exists, so the stale mismatched-epoch-budget run was deleted rather than
  left to confuse the controlled comparison — commits `a7ca690`/`be0294b`).
- **DC+TopK10 loss variant dropped (2026-08-19 night), a real negative
  finding, not just an incomplete run.** `nnUNetTrainerDCTopk10` was written
  as a hedge after nnU-Net's stock pure-TopK10 loss collapsed to an
  all-empty prediction on the real 500-epoch run — Dice's overlap term was
  meant to anchor against that. Its overfit-sanity-gate (100 epochs / 6
  cases) caught the same collapse before any real GPU time was spent:
  Pseudo Dice stayed at exactly 0.0 for all 100 epochs while train_loss kept
  moving. Confirmed not a wiring bug (loss construction matches nnU-Net's
  own stock pattern; a standalone synthetic-batch test produced a sane
  finite gradient) — a real optimization-dynamics failure on this dataset's
  extreme foreground imbalance. Decision: dropped, not retried — report
  both TopK-family collapses as a negative result next to the other
  loss-variant comparisons. See `CLAUDE.md` "DC+TopK10 dropped" entry for
  the full evidence trail.
- **ResEnc-M architecture run in progress**, `nnUNetTrainerBaseline_500epochs`
  on `nnUNetResEncUNetMPlans` — isolates the architecture-only effect vs.
  plain `baseline-500` (see `CLAUDE.md` "Switch to nnU-Net's ResEnc planner"
  future-consideration entry). Launched 2026-08-19 22:02 via
  `workspace/queue_resencm_baseline500.sh`, which also chains predict →
  evaluate (val + test) → aggregate → plot once training finishes — no
  action needed to pick up the results.

**Not yet done, next up:** run `dice_vs_threshold.py` and
`probability_histogram.py` against the real `predTs_prob/` exports for at
least `baseline-500` (and ideally every condition, for a Methods-section
threshold-tuning figure); decide whether the connected-component "no-op
wins" finding changes plans for prediction ensembling (unlikely, but
ensembling should be evaluated on its own merits regardless).

## Training-tips checklist review, and the two-night plan it drove

Cross-checked against the course's "General Training Tips" 6-section
practical checklist (sanity checks/diagnosis, data representation,
preprocessing discipline, loss/optimization, checkpoints/ensembles,
debugging workflow — the 7th section, generative/fine-tuning, doesn't apply
here). Source: `General Training Tips - Practical Checklist.pdf`.

**Already covered, in several places exceeding the checklist's bar:**
diagnosis discipline (`check_status.sh`; caught the 2026-08-17
sampling-collapse purely from Dice pinned at 0.0 while loss kept moving, see
`CLAUDE.md`); deliberate data representation (full-volume `3d_fullres`,
patch-based); preprocessing discipline (per-volume z-score, gzip integrity
checks, header-name metadata parsing after a real site-level column-swap
bug); patient- *and* site-level splitting (`test_ood` holds out whole
sites, beyond the checklist's "no patient leakage" ask); class imbalance
handled and iterated on (lesion-size-bin sampling, with a documented
bug-fix history); loss selection studied as a controlled experiment across
6 variants; checkpointing plus run-fingerprint provenance, beyond the
checklist's ask.

**Gaps the checklist surfaced, and where each stands now:**

| Gap | Status |
|---|---|
| No prediction ensembling | Still open — see "not done" below |
| LR/weight decay/optimizer are untouched defaults | Deferred, see Future work |
| No overfit-tiny-subset sanity test | **Done** — `nnUNetTrainerOverfitCheck.py` + `sanity_overfit_check.sh` |
| Augmentation strategy is default, not deliberate | **Done** — `wideaug-500` trained/predicted/evaluated, see below |
| No EMA/SWA weight averaging | Deferred, see Future work — low priority |

**The two-night plan this produced (2026-08-18 night one, 2026-08-19 final
run) — status: complete.** All items below finished; kept in the past tense
as a record of what the plan actually covered and how it played out. See
"Status as of 2026-08-19" above for the outcomes/findings this produced.

Baseline/focal-tversky/tversky-mild/sampling (plain 4:2:1) were already done
at 500 epochs on `Dataset002_ATLAS` before either of these two nights.

- **Night one, landed:** `baseline-wideaug-500`
  (`nnUNetTrainerWideAugBaseline_500epochs`), widened brightness/contrast/
  gamma vs. plain `baseline-500` — a clean A/B isolating augmentation's
  effect on the `test_id`/`test_ood` gap, chosen over a domain-adversarial
  approach as the cheaper first lever (see `CLAUDE.md` "Cross-center
  generalization" entry). Trained/predicted/evaluated; overfit-sanity-gate
  run against the new trainer before trusting the real launch — the check
  that would have caught the 2026-08-17 collapse in minutes instead of 96
  epochs, applied preemptively here.
- **Night one, also landed:** site-weighted (macro, mean-of-per-site-means)
  split summary in `aggregate_results.py`, guarding the `test_ood` ≥
  `test_id` anomaly against being an artifact of a few high-count sites
  dominating the pooled mean.
- **Night two (2026-08-19), landed:** `sampling-pow` (p=0.5, sqrt-dampened
  correction), originally deprioritized behind the `wideaug` A/B and only
  run at 250 epochs / `Dataset001` — ported to 500 epochs / `Dataset002`
  (`nnUNetTrainerLesionAwareSamplingPow_500epochs_full`), trained, predicted,
  and evaluated; the mismatched-epoch-budget 250-epoch run was removed. See
  "Status as of 2026-08-19" above.
- **Night one, also landed: connected-component post-processing.**
  `evaluation/postprocess_predictions.py` — drops predicted connected
  components below `--min-voxels` (26-connected by default, matching
  `lesion_wise_f1`), writes filtered masks to a new directory, never
  touches the source predictions (hard-guarded: refuses if `--out-dir`
  is or contains `--pred-dir`). Motivation and a real early result: a
  manual test on `DiceOnly_250epochs`/`Dataset001` (258 test cases,
  ground truth recovered from the archived raw dataset) found
  `--min-voxels 10` left Dice essentially unchanged (+0.0002) and
  improved lesion-wise F1 (+0.035, fewer phantom predicted lesions), but
  made mean HD95 *worse* (+0.35mm) — some "small" components turned out
  to be genuine diagonally-attached satellite lesion fragments, not
  noise, so a single hand-picked threshold isn't safe without tuning.
  Queued to search this properly and ran fully unattended overnight
  (2026-08-18→19), behind the GPU pipeline so it never contended for GPU
  or CPU with real training. Completed — see "Status as of 2026-08-19"
  above for the result (the no-op control won the search):
  - `run_postprocess_grid.sh` — 20 combos (`min_voxels` ∈
    {1,2,3,4,5,6,8,10,15,20} × `connectivity` ∈ {1,3}; `min_voxels=1` is
    the raw/no-op control) filtered + scored **only** against
    `baseline-500`'s own internal validation predictions (120 cases,
    `fold_0/validation/` vs `labelsTr`) — deliberately never against
    `predTs` (test_id+test_ood) during the search itself, to avoid tuning
    on the held-out test set. Auto-selects the combo with the lowest val
    mean HD95 (dice as tiebreaker), writes `selected_combo.json`, then
    applies that one frozen combo to `predTs` (250 cases) vs `labelsTs`
    exactly once, alongside a raw/unfiltered `predTs` scoring for a direct
    before/after row.
  - `queue_postprocess_after_samplingpow.sh` — waits for
    `/tmp/queue_samplingpow.sh` (sampling-pow-500 train → predict →
    evaluate → aggregate → plot, itself queued behind `wideaug-500`) to
    fully exit before starting the grid, per explicit request — keeps the
    whole night's GPU pipeline and this CPU-only grid from ever running
    concurrently, even though the grid itself never touches the GPU.
  - Results, confirmed present after wake-up:
    `workspace/postprocess_grid/baseline500/grid_summary_val.csv` (all 20
    combos), `selected_combo.json` (winner + selection rule),
    `test_final/results_*_test.csv` (frozen before/after on held-out test).
    Every combo's output is self-describing (a `*_config.json` with its
    params next to its filtered masks) and indexed by `grid_manifest.csv`,
    so any folder is traceable back to its exact `(min_voxels,
    connectivity)` combination.
- **The final run has landed (2026-08-19)** — all 7 conditions are trained,
  predicted, and evaluated at 500 epochs / `Dataset002`; see "Status as of
  2026-08-19" above for what came out of it and what's still open.
- **Prediction ensembling** — code now exists (`ensembling/ensemble_val.py`
  + `analysis/{select_finalist_from_val,plot_finalist_selection,
  pca_model_redundancy}.py`); per-case-Dice-correlation/PCA analysis run
  2026-08-20 over all 9 val results (see "Finalist-selection /
  ensembling-candidate analysis" below) recommends WideAugBaseline +
  Baseline(ResEncM) + FocalTversky + LesionAwareSamplingPow as the ensemble
  candidate set. Still pending: exporting `predVal_prob/` for that set and
  running `ensemble_val.py` itself to confirm on real voxel-probability
  correlation + actual ensembled scores, not just the case-Dice proxy.

## Report deliverables — what's ready vs. what needs building

Per course guidelines (`Project_Guidelines_2026.pdf`): Methods section
justifying architecture/loss/sampling choices; Experiments section with
comparisons, ablations, hyperparameters, visualization, failure-mode
discussion; graphs/tables/figures throughout.

**Already safe — assemble later, no rerun needed:**
- Per-epoch train/val loss, pseudo-Dice, LR, epoch time — nnU-Net writes
  `training_log_*.txt` + `progress.png` per run automatically.
- Full hyperparameter/config provenance per run — `debug.json` +
  `isles26_fingerprint.json` + `run_manifest.json` (see `CLAUDE.md`
  "Run-identity fingerprinting" entry).
- Held-out predictions (`predTs/*.nii.gz`) saved to disk for every finished condition.
- Case-level metrics + overall/by-size/by-split/by-center/volume-scatter figures — already implemented (`evaluation/aggregate_results.py`, `analysis/plot_results.py`).

**Done (2026-08-19):**
1. Training-curve parser — `analysis/plot_learning_curves.py`, built and run
   (`figures/learning_curves_all_trainers.png`,
   `figures/lr_schedule_all_trainers.png`,
   `figures/train_val_overlay_all_trainers.png`; see "Status as of
   2026-08-19").
4. Sampling-weight distribution figure for `sampling-pow` is now unblocked —
   `case_metadata_pow_p05_full.csv` (963 cases, `Dataset002` scale) exists;
   the figure itself still needs to be generated.

**Still needs a script (no GPU required, doesn't block training):**
2. Split-aware per-center figure: `save_by_center` currently pools
   train/val/test_id/test_ood per site together; needs faceting by `split` (at minimum `test_id` vs. `test_ood` per center) to show whether the `test_ood` ≥ `test_id` anomaly is site-specific.
3. Qualitative overlay figures (input / ground truth / prediction / error map — a few representative + failure cases). Predictions are already saved, so safe to build anytime, but don't leave it to report week.

**New exploratory scripts, built 2026-08-19, not yet run against real data:**
`analysis/dice_vs_threshold.py` (Dice vs. foreground-probability threshold,
checks whether nnU-Net's implicit fixed 0.5 cutoff is actually optimal) and
`analysis/probability_histogram.py` (predicted-probability distribution by
true GT class) — both consume the newly-saved `predTs_prob/` softmax exports
(see "Status as of 2026-08-19"). CPU-only, no rerun of inference needed.

**Done (2026-08-18):** before/after augmentation example figures --
`analysis/save_augmentation_examples.py`, run against a real preprocessed
case (`ATLAS_r001s001_ses1`, large lesion) with the exact rotation/patch/
mirror config nnU-Net computed for tonight's actual `baseline-wideaug-500`
run. Three PNGs under `workspace/figures/augmentation_examples/`:
- `augmentation_comparison_*` — honest, as actually seen during training
  (each intensity transform only fires ~15-30% of the time, so a single draw
  often shows little difference — expected, not a bug).
- `augmentation_forced_intensity_*` — illustrative only, probability forced
  to 1 so the stock-vs-widened range difference is actually visible; not
  what training runs.
- `augmentation_context_*` — whole preprocessed volume with the 128^3
  training-patch region boxed, since any single patch is smaller than the
  full head and (for a large, off-center lesion) won't show the whole brain
  by itself.
This was genuinely time-sensitive: the augmentation *config* itself isn't
saved anywhere by nnU-Net during training, unlike the per-epoch curves above.
Now captured in both the trainer's code and this reusable script — no more
urgency, can be regenerated for other cases anytime.

**Done (2026-08-18→19):** connected-component post-processing grid search —
see the "Tonight, also landed" entry above for the script names and method,
and "Status as of 2026-08-19" above for the result (auto-selected winner is
the no-op control; report as a real negative finding). Gives the report a
real Methods-section post-processing step with its own before/after
Experiments-section row.

**Explicitly deferred:** prediction ensembling and anything built on it —
prerequisite probability exports now exist (`predTs_prob/`, all 7
conditions) but no ensembling code has been written yet.
Domain-adversarial training (gradient-reversal on encoder features) — a
stretch idea for the report's Conclusion/future-work section only, not in
scope for the final run; see `CLAUDE.md` for why augmentation was tried
first.

## Resource fallback

If GPU time is limited, reduce the number of folds — but keep the exact
same fold for every method (see the controlled-comparison rules above).

## Report caveats to state explicitly (not fixes — no time left for a follow-up run)

- **Single fold.** Useful for iteration, insufficient alone for strong
  statistical claims. Fold and split are identical across every method,
  which is what makes the comparison valid despite this.
- **`test_ood` ≥ `test_id` Dice in every condition so far** — the opposite
  of ISLES'26's motivating premise (a cross-center generalization gap).
  Could be genuine, could be an artifact of which 6 sites ended up in the
  locked OOD set (`R005, R008, R027, R029, R042, R070`); the site-weighted
  summary above and the still-missing per-site `test_ood` figure are the
  next steps if pursued after the final run.
- **Inverse-lesion-volume sampling can heavily overweight the smallest
  cases** — inspect the resulting probability distribution before
  interpreting the sampling experiment (see the 2026-08-17 sampling-collapse
  bug in `CLAUDE.md` for why this matters concretely).
- **ATLAS discovery patterns still need confirmation against the exact
  dataset download** — not yet verified as of this writing; check before
  relying on case-discovery counts in the report.

## Ensembling investigation concluded: no significant improvement over the best single model (2026-08-21)

Closes out the "Finalist-selection" investigation below with real, final
val + held-out-test numbers. **Bottom line: ensembling does not produce a
statistically significant improvement over the single best model on this
dataset — confirmed independently on val and on test_id/test_ood, across
every metric and every size/split breakdown.** Report as a real
negative/inconclusive finding for the Experiments section, same category as
the connected-component post-processing "no-op wins" result above.

**Code restructured first, at the user's explicit request** — `analysis/`
now holds only read-only correlation/PCA scripts (twin probability-level
scripts added: `voxel_probability_correlation.py`, `plot_voxel_probability_
correlation.py`, `pca_probability_redundancy.py`, plus a shared
`predval_dirs.py`), while `ensembling/` holds only real, expensive work
(`ensemble_val.py` — build + score a chosen combo on val, no plotting;
new `ensemble_test.py` — the one-time, deliberate held-out scoring step for
a chosen combo, see below).

**Two real bugs found and fixed while wiring this up:**
- `predval_dir()`/new `predtest_dir()` needed a fallback: standard-plans
  `Baseline` had its val probabilities in nnU-Net's automatic `validation/`
  folder (never in `export_val_probabilities.sh`'s trainer list), and both
  ResEncM-plans runs had their *test*-set probabilities sitting in the plain
  `predTs/` folder (saved on the original predict call) rather than a
  dedicated `predTs_prob/` export. Verified real (not stale): correct case
  counts, correct `.npz` schema, newer than `checkpoint_final.pth`.
- `score_combo()`'s per-case loop called `evaluate_case` with no exception
  handling — one real geometry-mismatch case (`ATLAS_r032s013_ses1`) crashed
  the *entire* combo-scoring run before this fix, instead of being skipped
  like `compute_metrics.py`'s own CLI already does. Fixed to match.

**Primary metric switched `hd95_mm` → `dice`, project-wide** (see
`CLAUDE.md` "Primary metric switched to Dice" for the full why: Dice is
always defined — HD95 is NaN on any empty-lesion case — lower-variance on a
val/test set this size, and the metric every comparable report leads with).
This **changed the recommended ensemble combo** from the 2026-08-20 pick
below — re-ran `select_finalist_from_val.py`/`pca_model_redundancy.py`/
`pca_probability_redundancy.py`/`plot_finalist_selection.py` under the new
default and got real, different answers, not just relabeled ones:
- New val ranking (dice): `WideAugBaseline (ResEncM)` #1 (0.6553),
  `Baseline (ResEncM)` #2 (0.6550, statistically tied), `FocalTversky` #3,
  `Baseline` #4, `WideAugBaseline` #5 ... `LesionAwareSamplingPowCurriculum`
  #8 (0.6173) now beats `LesionAwareSamplingPow` #9 (0.6127) — a real flip
  from the hd95_mm ranking, and consistent with what the voxel-probability
  correlation already suggested (Curriculum was the more complementary of
  the two, even under the old ranking).
- Redundancy-cluster "keep" picks flip in **both** clusters under dice:
  `WideAugBaseline (ResEncM)` now wins the ResEncM cluster (was `Baseline
  (ResEncM)`); `Baseline` now wins the standard-plans cluster (was
  `WideAugBaseline`).
- Voxel-probability-correlation analysis is metric-independent, so it's
  unchanged: max correlation found anywhere is still 0.937 (`Baseline` vs.
  `WideAugBaseline`), all 9 trainers remain singletons at the r=0.95
  redundancy cut — doesn't independently justify dropping anyone, but
  doesn't contradict the case-Dice-based narrowing either.
- **New recommended/tested combo: `Baseline + WideAugBaseline (ResEncM) +
  FocalTversky + LesionAwareSamplingPowCurriculum`** (replaces the
  2026-08-20 pick, `WideAugBaseline + Baseline(ResEncM) + FocalTversky +
  LesionAwareSamplingPow`, below).

**Real val-side scoring** (`ensemble_val.py`, all 7 combos — 6 pairs + the
full 4-way average, real softmax-averaging + real metric scoring, not a
proxy): best single model `WideAugBaseline (ResEncM)` (dice 0.6553); **no
combo significantly beats it** (p = 0.14–0.97 across all 7); one combo
(`WideAugBaseline (ResEncM)+LesionAwareSamplingPowCurriculum`) is
significantly *worse* (p < 0.001). `workspace/results/finalist_selection/
ensemble_summary_val.csv`.

**Real held-out test scoring** (`ensembling/ensemble_test.py`, new script —
one-time, deliberate; reuses existing single-model `results_test.csv` rows
rather than re-predicting them; writes a report table broken down by
overall / split (test_id, test_ood) / **size_bin (small, medium, large)** —
size_bin matters as much as split here, per `docs/reference/`'s "Match
Architecture to Pathology Size": small lesions are the case a generic
pipeline is most likely to lose signal on, and "best pooled Dice" can hide
"best where it's hardest"). Compared the new combo against `Baseline` and
`WideAugBaseline (ResEncM)` (the primary-metric anchor, always included),
249/250 held-out cases scored, real paired-bootstrap p-values (a two-sided
bootstrap p-value was added to `paired_bootstrap_vs_top`, reused by both
scripts):

| breakdown | metric | ensemble | Baseline | WideAugBaseline (ResEncM) | p-value |
|---|---|---|---|---|---|
| overall (249) | dice | 0.6549 | 0.6530 | 0.6552 | 0.971 |
| overall | hd95_mm | **16.41** | 17.94 | 17.09 | 0.787 |
| overall | lesion_f1 | **0.664** | 0.637 | 0.657 | 0.483 |
| test_id (120) | dice | 0.6360 | 0.6278 | 0.6393 | 0.429 |
| test_ood (129) | dice | 0.6725 | 0.6764 | 0.6700 | 0.677 |
| large (78) | dice | 0.8252 | **0.8325** | 0.8175 | 0.142 |
| large | hd95_mm | **7.153** | 7.283 | 7.242 | 0.781 |
| medium (84) | dice | 0.6325 | 0.6270 | **0.6326** | 0.958 |
| medium | hd95_mm | **16.835** | 18.837 | 18.101 | 0.765 |
| small (87) | dice | 0.5238 | 0.5172 | **0.5314** | 0.342 |
| small | hd95_mm | **24.591** | 26.831 | 25.142 | 0.941 |
| small | lesion_f1 | **0.647** | 0.627 | 0.643 | 0.801 |

(full 18-row table, every metric x every breakdown, in `workspace/results/
runs/ensemble_Baseline+WideAugBaseline (ResEncM)+FocalTversky+
LesionAwareSamplingPowCurriculum/report_table.csv`)

**No row anywhere reaches significance** (p ranges 0.14–0.97 across all 18
rows). The one consistent pattern: the ensemble is numerically the best of
the three on HD95 and lesion-F1 in **every single breakdown** (overall,
both splits, and all three size bins) — but never enough to clear
significance given the sample sizes (78–249 cases per breakdown). On Dice
specifically there's no consistent winner (Baseline wins large,
WideAugBaseline(ResEncM) wins medium and small, the ensemble wins none) —
also with no significant separation. Even on the small-lesion subgroup
specifically, the ensemble shows no Dice advantage over the single best
model.

**Final single-model choice for the report: `WideAugBaseline (ResEncM)` —
best-ranked among a statistically tied group, not a clean outright winner.**
Reasoning to carry into the Methods/Experiments writeup:
- **In favor:** best point estimate on val (dice 0.6553); held up on the
  real held-out test set (won pooled dice, 0.6552 vs. the ensemble's 0.6549
  and `Baseline`'s 0.6530; won dice specifically on the medium and small
  size bins); no ensemble combo — including one built specifically to try
  to beat it — showed a significant improvement over it, on val or test.
  It also stacks the two interventions with the clearest independent
  motivation in this project (widened augmentation from the `wideaug` A/B,
  and the ResEnc-M architecture upgrade) — a clean story for the report.
- **Caveat to state explicitly, not bury:** on val it is **statistically
  tied with 5 of the other 8 trainers** (`Baseline (ResEncM)`,
  `FocalTversky`, `Baseline`, `WideAugBaseline` standard-plans,
  `TverskyMild`) — only the 3 LesionAwareSampling variants are
  significantly worse. So this is "best-ranked among a tied group," not
  "significantly best" — report it that way.
- **Cost caveat:** ResEnc-M is a materially larger/more expensive
  architecture than the standard-plans baseline (see `CLAUDE.md` "Switch to
  nnU-Net's ResEnc planner"). Plain `WideAugBaseline` (no ResEncM) is
  statistically indistinguishable from it and much cheaper to train/run —
  name it explicitly as the cheaper, equally-defensible alternative if
  compute cost matters for the report's conclusions, rather than silently
  defaulting to the pricier architecture on a non-significant point-estimate
  margin.
- **Metric caveat:** on HD95 and lesion-F1 specifically (not the primary
  metric, but still reported), the 4-model ensemble was numerically ahead of
  `WideAugBaseline (ResEncM)` in every single breakdown above (though never
  significantly) — worth a sentence in the writeup if the report's framing
  cares about lesion-detection completeness (lesion-F1) or worst-case
  boundary error (HD95), not just Dice.

**Next step: not yet decided — pending discussion** (e.g. whether to try
the earlier-flagged Pow-vs-Curriculum swap as an ablation, whether this is
enough to write up as the final ensembling result, or whether a genuinely
different combo/method is worth one more real val-side check before
closing this out).

## Finalist-selection / ensembling-candidate analysis (2026-08-20) — SUPERSEDED, see entry above

The recommended combo and hd95_mm-primary ranking below were superseded by
the 2026-08-21 entry above once the primary metric switched to Dice. Kept
as a historical record of what was concluded at the time, not retroactively
rewritten.

Ran `analysis/finalist_selection/select_finalist_from_val.py`,
`analysis/finalist_selection/plot_finalist_selection.py`, and
`analysis/finalist_selection/pca_model_redundancy.py` over all 9 trained val
results (the original 7 500-epoch conditions plus the two ResEncM-plans
runs, `baseline` and `wideaug`, both finished since the "Status as of
2026-08-19" entry above). Output: `workspace/results/finalist_selection/*.csv`
and `workspace/figures/finalist_selection/*.png`.

**Bug fixed first, not cosmetic:** `discover_runs()` (in
`select_finalist_from_val.py`) keyed runs by trainer class name only. Since
`nnUNetTrainerBaseline_500epochs` and `nnUNetTrainerWideAugBaseline_500epochs`
each now have two runs (standard plans vs. `nnUNetResEncUNetMPlans`), the
ResEncM run was silently overwriting its standard-plans sibling in the dict —
2 of 9 runs would have vanished from every downstream comparison with no
error. Fixed to key by trainer **+ plans** (non-default plans get a
`(PlansName)` suffix, e.g. `(ResEncM)`), and to raise loudly instead of
silently dropping on any future collision. All 9 runs share the identical 119
val cases, so the paired comparison is valid.

**Ranking (hd95_mm primary, val):** WideAugBaseline leads, but FocalTversky,
Baseline(ResEncM), TverskyMild, and plain Baseline are all statistically tied
with it (paired bootstrap, no significant separation from #1). Only the 3
LesionAwareSampling variants (plain, Pow, PowCurriculum) are significantly
worse.

**Correlation/PCA says there are ~4 real axes of behavior among the 9, not 9
independent models** — the basis for an ensembling recommendation:
- Standard-plans "generic" cluster (r≈0.96–0.98, effectively redundant):
  WideAugBaseline, Baseline, LesionAwareSampling → keep only **WideAugBaseline**
  (best-ranked).
- ResEncM-plans cluster (r=0.95 with each other, but its own distinct PCA
  region — a genuinely different architecture/plans axis): Baseline(ResEncM),
  WideAugBaseline(ResEncM) → keep only **Baseline(ResEncM)** (best-ranked;
  also the single lowest pairwise correlation found anywhere, r=0.741 vs.
  TverskyMild — the most complementary pair on this dataset).
- Tversky-loss family (r=0.95 with each other — borderline-redundant even
  though the 0.95 cutoff didn't formally merge them): FocalTversky,
  TverskyMild → keep only **FocalTversky** (ties for #1, marginally ahead).
- Sampling-correction family (its own distinct PCA axis, but both
  significantly worse than the top on hd95_mm): LesionAwareSamplingPow,
  LesionAwareSamplingPowCurriculum → keep only **LesionAwareSamplingPow**
  (better of the two: hd95 21.9mm vs. 22.1mm).

**Recommended ensemble candidate set: `WideAugBaseline + Baseline(ResEncM) +
FocalTversky + LesionAwareSamplingPow`** — one representative per distinct
error-pattern cluster, chosen to maximize complementary errors rather than
just raw solo rank. This differs from `ensemble_val.py`'s current hardcoded
`DEFAULT_TRAINERS` (still the pre-ResEncM 5: WideAugBaseline, FocalTversky,
TverskyMild, LesionAwareSamplingPow, LesionAwareSamplingPowCurriculum) and
from `export_val_probabilities.sh`'s `TRAINERS` array (same stale 5) —
**not yet updated to match this recommendation**, pending the raw-probability
follow-up below.

**Caveat — this is a proxy, not the final call.** All of the above is built
from per-case *binary-mask* Dice correlation (`load_all`'s `dice` column),
not the actual voxel-level softmax probabilities the real ensemble would
average. `ensemble_val.py` exists precisely to check this proxy against (a)
per-voxel probability correlation and (b) real ensembled Dice/HD95/lesion-F1
scored against the best single model — run that (needs `predVal_prob/`
exported per trainer first, via `export_val_probabilities.sh`) before
finalizing which trainers actually go into the shipped ensemble.

## Post-processing grid search rerun under Dice-primary selection: wideaug_resencm (2026-08-21)

Closes out the "rerun under `--selection-metric dice_mean`" item that was
still open in "Future work" below (now struck through there) — the primary
metric switched from hd95_mm to Dice on 2026-08-20 (see `CLAUDE.md` "Primary
metric switched to Dice"), so the original `baseline500` grid's "no-op wins"
conclusion needed re-checking under Dice selection, not assumed to carry
over. Run against `WideAugBaseline (ResEncM)`, the current best single model
per the finalist-selection ranking above (not `baseline500` again).

**Method, same discipline as the original grid** (`evaluation/
postprocess_predictions.py` + `evaluation/summarize_postprocess_grid.py`,
driven by `run_postprocess_grid.sh`): connected-component filtering swept
over `min_voxels` x `connectivity`, selected by lowest val `dice_mean`
(hd95_mean tiebreak) against the 119-case internal validation split only
(`fold_0/validation` vs `labelsTr`) — never against `test_id`/`test_ood`
during selection — then the one frozen winning combo applied exactly once
to the held-out test set for a before/after row. Log:
`workspace/logs/postprocess_grid_wideaug_resencm_20260821_003620.log`.
Outputs: `workspace/predictions/postprocess_grid/wideaug_resencm/
grid_summary_val.csv`, `selected_combo.json`,
`test_final/results_{raw,cc15_conn2}_test.csv`.

**Selected combo: `cc15_conn2`** (drop components <15 voxels, 18-connected).
Same qualitative "no-op wins" shape as the original hd95-primary
`baseline500` grid, now confirmed under Dice-primary selection too:

| | val (n=119) Dice | val HD95 | val lesion F1 | test (n=249) Dice | test HD95 | test lesion F1 |
|---|---|---|---|---|---|---|
| raw (no-op) | 0.6553 | 20.006 | 0.651 | 0.6552 | 17.095 | 0.6570 |
| **cc15_conn2 (selected)** | **0.6560** | 20.081 | 0.6843 | 0.6529 | 16.843 | **0.6927** |

Val Dice improvement is +0.0007 — noise-level, same order of magnitude as
every other combo in the grid (all 20+ combos land within ≤0.0007 Dice of
the no-op; see `grid_summary_val.csv`). On the frozen test check, Dice
actually goes the *other* way (-0.0023) while HD95 improves slightly
(-0.25mm) and lesion F1 improves more (+0.036) — components small enough to
prune are mostly spurious false-positive lesions, not volume that matters
for Dice, so pruning them helps lesion-wise detection without moving the
voxel-overlap metric either direction reliably.

**Two illustrative cases (`test_final/results_{raw,cc15_conn2}_test.csv`,
matched by `case_id`):**
- **Helped:** `ATLAS_r050s001_ses1` — raw predicted 2 lesion components (1
  true positive + 1 spurious ~10-voxel blob), lesion F1 0.667; after
  filtering, only the true-positive component survives, Dice 0.737 -> 0.848
  (+0.112) and lesion F1 0.667 -> 1.0. This is the mechanism the grid is
  supposed to catch — a small phantom blob dragging both metrics down.
- **Hurt (worst case in the whole test set):** `ATLAS_r053s042_ses1` — raw
  predicted exactly 1 lesion component, smaller than the 15-voxel cutoff.
  Filtering removes it entirely, leaving an empty prediction against a
  non-empty ground truth: Dice 0.714 -> **0.0**, lesion F1 1.0 -> **0.0**.
  This is the same failure mode flagged in the original `baseline500`
  finding (`CLAUDE.md`/"Night one" entry below) — some small components are
  genuine tiny lesions, not noise, and a size-only cutoff can't tell the
  difference. 92 of 249 test cases changed at all under `cc15_conn2` (most
  changes much smaller than these two extremes); this is the largest single
  swing in either direction.

**To be decided: whether to actually ship `cc15_conn2` post-processing, or
report raw predictions as final.** Arguments either way, not yet resolved:
- *For post-processing:* consistent, reproducible lesion-F1 gain (+0.033
  val / +0.036 test) with a Dice cost inside the run-to-run noise band; if
  the report's Discussion cares about false-positive lesion count (e.g.
  clinical usability framing), this is a real, defensible improvement to
  ship.
- *Against:* the Dice movement is a coin flip in direction (+val, -test),
  and the `r053s042` failure mode shows the tradeoff isn't free — the same
  cutoff that removes phantom blobs will occasionally remove a real,
  correctly-detected small lesion outright (Dice/F1 -> 0 for that case).
  Given Dice is now the project's primary metric (see `CLAUDE.md`), and the
  grid was explicitly re-run to check exactly this, a defensible default is
  "no-op" (report post-processing as an investigated-and-rejected step,
  same framing as the original `baseline500` finding) unless lesion-wise
  detection quality is independently important to the report's framing.
- Not yet run: the equivalent Dice-primary rerun for `baseline500` itself
  (the original grid target) — only `wideaug_resencm` has been redone under
  the new selection metric so far. Worth doing before finalizing the
  report's post-processing section, for completeness across conditions, not
  because the conclusion is expected to differ.

## Future work (beyond this report)

- ~~Rerun the connected-component post-processing grid search under
  `--selection-metric dice_mean`~~ **Done for `wideaug_resencm`, 2026-08-21**
  — see the dedicated section above. `baseline500`'s own Dice-primary rerun
  is still outstanding (noted in that section's last bullet).
- **`baseline500`'s own Dice-primary post-processing grid rerun** — still
  outstanding; only `wideaug_resencm` has been redone under
  `--selection-metric dice_mean` so far (see the dedicated section above).
  Same val-only discipline applies (`fold_0/validation/` vs. `labelsTr`,
  never test_id/test_ood).
- **Decide whether to ship `cc15_conn2` post-processing or report raw** —
  see "To be decided" in the post-processing section above; not yet
  resolved.
- **Per-condition threshold tuning** (0.5 → best-found, via
  `dice_vs_threshold.py`) and probability-calibration inspection (via
  `probability_histogram.py`) across all 7 conditions, now that
  `predTs_prob/` exists for each — natural next exploratory pass once the
  report's fixed deliverables are locked, since it's the same probability
  data prediction ensembling will need.
- **Prediction ensembling — concluded (2026-08-21).** See "Ensembling
  investigation concluded" above: no statistically significant improvement
  over the single best model, confirmed on both val and held-out test.
  Report as a real negative/inconclusive finding; next step (if any) is a
  discussed-not-decided ablation, not a rerun of what's already confirmed.
- **Domain-adversarial training** (gradient-reversal domain classifier on
  encoder features) — only worth revisiting if the `test_id`/`test_ood` gap
  is still meaningful after the `wideaug` result. See `CLAUDE.md` for the
  reasoning against doing this first.
- **Curriculum (annealed-p) power-law sampling** — new trainer
  `sampling-pow-curriculum-500` (2026-08-19, see `CLAUDE.md`), p ramped 0→1
  in 20 steps of 25 epochs instead of the fixed p=0.5 `sampling-pow-500`
  already used. Built and CLI-verified (`--print-only`) but not run —
  no GPU time left for a follow-up 500-epoch condition before the report is
  due.
- **ResEnc planner (M or L)** instead of the default nnU-Net planner — an
  architecture change, not a tuning knob; would need to apply uniformly
  across all conditions to stay controlled. See `CLAUDE.md` "Future
  considerations" for GPU budget notes.
- LR / weight decay / optimizer choices — untouched nnU-Net defaults,
  never revisited as a deliberate lever.
- EMA/SWA weight averaging — not implemented; low priority since nnU-Net
  already tracks a pseudo-Dice EMA for checkpoint selection.
- **Learned post-processing network** (discussed 2026-08-19, not
  implemented) — a second, sequential model that decides which predicted
  connected components to keep, instead of one hand-picked
  `--min-voxels` threshold. Motivated directly by the finding above: a
  size-only cutoff can't distinguish real satellite lesion fragments from
  spurious blobs, and the grid search only covers a 1D/2D hyperparameter
  space (size × connectivity), not the richer per-component signal
  (predicted-probability confidence, shape, distance to other components)
  a learned decision could use. Three options discussed, cheapest first:
  1. **Tabular classifier over per-component features** (voxel count,
     mean/max softmax probability, shape/compactness, distance to nearest
     other component) — logistic regression or gradient-boosted trees,
     trained on val-set components labeled by real overlap with ground
     truth. CPU-only, no GPU contention, buildable without touching the
     training pipeline — the only one of the three realistic to attempt
     before the report deadline, if pursued at all.
  2. **Small 3D patch CNN classifier** centered on each component
     (image + probability-map crop → keep/discard) — the "false-positive
     reduction network" pattern from candidate-based lesion/nodule
     detection (e.g. LUNA16-style pipelines). More powerful than (1), but
     needs real GPU training time and its own train/val discipline.
  3. **Full cascade/refinement segmentation network** — a second full
     model taking `[image, stage-1 predicted mask]` as input and
     outputting a refined mask (nnU-Net has this built in as
     `3d_cascade_fullres`). Most powerful, but GPU-cost comparable to
     adding a whole extra condition to the study.
  Whichever is pursued later: it must be trained/tuned on `val`-derived
  components only, same discipline as the grid search above — never on
  `test_id`/`test_ood` components, or it leaks into the generalization
  claim the whole `test_ood` split exists to protect.
