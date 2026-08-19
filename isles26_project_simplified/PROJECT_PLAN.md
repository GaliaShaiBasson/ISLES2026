# Project plan and status

This file is *what's done, what's running, and what's left* before the
report is due. See `README.md` for setup and the exact CLI commands, and
`CLAUDE.md` for *why* each deviation from nnU-Net/project defaults was made
and for the full decisions log.

## Phase order and controlled-comparison rules

Split → prepare → preprocess → debug smoke test → baseline → loss study →
sampling → evaluate/aggregate/plot (see `README.md` for the commands). A few
rules apply across all of it, not obvious from the commands alone:

- Use the same fold, configuration, and epoch budget (250, later 500 — see
  `CLAUDE.md`) for every real comparison — changing more than one axis at a
  time makes the comparison uninterpretable.
- Don't proceed past the split step until `broken_cases.csv` is small/stable
  (i.e. the raw-data upload has actually finished).
- Before trusting a `sampling` run, inspect
  `workspace/case_metadata.csv`'s `sampling_weight` distribution.
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
- **Prediction ensembling** (softmax-averaging across finished checkpoints)
  remains deferred — per-case probabilities are now saved for every
  condition (`predTs_prob/`, see above), so the prerequisite data exists;
  still no code in `evaluation/` for it yet.

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

## Future work (beyond this report)

- **Per-condition threshold tuning** (0.5 → best-found, via
  `dice_vs_threshold.py`) and probability-calibration inspection (via
  `probability_histogram.py`) across all 7 conditions, now that
  `predTs_prob/` exists for each — natural next exploratory pass once the
  report's fixed deliverables are locked, since it's the same probability
  data prediction ensembling will need.
- **Prediction ensembling** (softmax-averaging across the 7 finished
  checkpoints) — zero GPU cost, prerequisite data (`predTs_prob/`) now
  exists for every condition; no code in `evaluation/` yet.
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
