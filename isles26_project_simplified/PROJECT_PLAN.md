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
| Augmentation strategy is default, not deliberate | **In progress** — tonight's `wideaug` run |
| No EMA/SWA weight averaging | Deferred, see Future work — low priority |

**The two-night plan this produced (2026-08-18 tonight, 2026-08-19 final run):**

Baseline/focal-tversky/tversky-mild/sampling (plain 4:2:1) were already done
at 500 epochs on `Dataset002_ATLAS` before either of these two nights.

- **Tonight — running now:** `baseline-wideaug-500`
  (`nnUNetTrainerWideAugBaseline_500epochs`), widened brightness/contrast/
  gamma vs. plain `baseline-500` — a clean A/B isolating augmentation's
  effect on the `test_id`/`test_ood` gap, chosen over a domain-adversarial
  approach as the cheaper first lever (see `CLAUDE.md` "Cross-center
  generalization" entry). Run the overfit-sanity-gate above against any
  newly-touched trainer before trusting a real launch on it — that's the
  check that would have caught the 2026-08-17 collapse in minutes instead
  of 96 epochs.
- **Tonight, also landed:** site-weighted (macro, mean-of-per-site-means)
  split summary in `aggregate_results.py`, guarding the `test_ood` ≥
  `test_id` anomaly against being an artifact of a few high-count sites
  dominating the pooled mean.
- **Not done, deprioritized in favor of the wideaug A/B:** `sampling-pow`
  (p=0.5, sqrt-dampened correction) only ever ran at 250 epochs /
  `Dataset001` — never ported to 500 epochs / `Dataset002` to match the
  rest of the study.
- **Tomorrow night is the real final run** — whatever it produces is what
  goes in the report, with no engineering time left afterward. Everything
  in "Report deliverables" below, except ensembling, must be ready
  *before* it lands.
- **Prediction ensembling** (softmax-averaging across finished checkpoints)
  is the one thing explicitly deferred until after tomorrow's run — zero
  GPU cost, no code in `evaluation/` yet, safe to build anytime against
  already-finished checkpoints.

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
- Held-out predictions (`predTs/*.nii.gz`) saved to disk for every finished
  condition.
- Case-level metrics + overall/by-size/by-split/by-center/volume-scatter
  figures — already implemented (`evaluation/aggregate_results.py`,
  `analysis/plot_results.py`).

**Needs a script that doesn't exist yet (no GPU required, doesn't block training):**
1. Training-curve parser: `training_log_*.txt` → per-epoch CSV → one
   comparison figure across conditions.
2. Split-aware per-center figure: `save_by_center` currently pools
   train/val/test_id/test_ood per site together; needs faceting by `split`
   (at minimum `test_id` vs. `test_ood` per center) to show whether the
   `test_ood` ≥ `test_id` anomaly is site-specific.
3. Qualitative overlay figures (input / ground truth / prediction / error
   map — a few representative + failure cases). Predictions are already
   saved, so safe to build anytime, but don't leave it to report week.
4. Sampling-weight distribution figure for `sampling-pow` — cheap
   Methods-section figure once `sampling-pow-500` metadata exists (blocked
   on the "not done" item above).

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

**Explicitly deferred:** prediction ensembling and anything built on it — the
one thing intentionally left for after tomorrow night's run.
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

- **Domain-adversarial training** (gradient-reversal domain classifier on
  encoder features) — only worth revisiting if the `test_id`/`test_ood` gap
  is still meaningful after the `wideaug` result. See `CLAUDE.md` for the
  reasoning against doing this first.
- **ResEnc planner (M or L)** instead of the default nnU-Net planner — an
  architecture change, not a tuning knob; would need to apply uniformly
  across all conditions to stay controlled. See `CLAUDE.md` "Future
  considerations" for GPU budget notes.
- LR / weight decay / optimizer choices — untouched nnU-Net defaults,
  never revisited as a deliberate lever.
- EMA/SWA weight averaging — not implemented; low priority since nnU-Net
  already tracks a pseudo-Dice EMA for checkpoint selection.
