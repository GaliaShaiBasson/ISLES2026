# Project review

## Overall assessment

The research structure is coherent: one baseline, a focused loss study, lesion-size-aware sampling, and stratified evaluation. The original operational layer was the weak point. It required several environment variables, duplicated Bash and PowerShell scripts, manual custom-trainer installation, and hand-built evaluation commands.



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

### Recommended before the next overnight run (2026-08-18, ~2-3h budget, 2-3 conditions max)

Ranked by value-per-minute and risk to an unattended run. Context: dataset002
500-epoch run had baseline/focal-tversky/tversky-mild done and sampling-500
finishing (epoch 418/500); going forward, runs are being capped at 2-3
conditions instead of running every variant every time.

1. **Overfit-tiny-subset sanity gate (30-45 min), do first.** A script/trainer
   variant that trains on a fixed 8-16 case subset until loss ≈ 0, run
   against whichever 2-3 trainers are picked for that night before the real
   launch. This is the one check that would have caught the Aug 17
   sampling-collapse bug hours earlier instead of after 96 wasted epochs —
   cheapest insurance against burning a whole night on a wiring bug in a
   newly-touched trainer. Directly closes gap 3 above.
2. **Prediction ensembling on already-finished checkpoints (45-60 min), zero
   risk, can run in parallel with training.** baseline-500 /
   focal-tversky-500 / tversky-mild-500 are already done and sampling-500 is
   about to finish — 4 real checkpoints already on disk. A simple
   softmax-averaging step in `evaluation/` over these gets free Dice on top
   of a study already paid for, without touching that night's GPU
   allocation at all. Directly closes gap 1 above.
3. **Widened-augmentation variant as one of the 2-3 nightly conditions
   (remaining time).** The one substantive open question from the "future
   considerations" section below — brightness/contrast/gamma ranges are
   still nnU-Net's generic mild defaults, not chosen for this dataset's
   cross-center generalization goal, and already scoped there as "try this
   first, cheap, one afternoon" before anything adversarial.

**Explicitly deferred**, not because they're wrong, but because they need
their own controlled validation before trusting on an unattended run and
don't fit a 2-3h prep window: LR/weight-decay/optimizer changes (gap 2), and
EMA/SWA (gap 5).

### Update (2026-08-18): two runs remain, not one — tonight and tomorrow night (final, presentation-bound)

baseline-500 / focal-tversky-500 / tversky-mild-500 / sampling-500 (plain
4:2:1) are already done at 500 epochs on the full `Dataset002_ATLAS`.
Tomorrow night's run is the real final one — results and everything needed
to report on them must be ready for the presentation right after it finishes,
so there's no engineering time left once it lands. That splits the remaining
work into two buckets:

**Tonight's run (after the 2-3h prep window, once `sampling-500` finishes):**

1. **Overfit-tiny-subset sanity gate, build first.** Run it against both new
   trainers below before launching either — neither has been exercised
   through real epochs yet, and this is exactly the check that would have
   caught the 2026-08-17 sampling-collapse bug before it wasted GPU time.
2. **`sampling-pow` at 500ep/dataset002.** Closes the question CLAUDE.md
   itself raised on 2026-08-17: plain 4:2:1 sampling showed "real cost,
   marginal/unclear benefit" by size bin, and `sampling-pow` (p=0.5,
   sqrt-dampened correction) was built as "the next informative data point"
   to tell "sampling doesn't help here" apart from "4:2:1 was too weak to
   show it." Implemented and verified, but only ever run at 250
   epochs/dataset001 — needs porting to 500ep/dataset002 to match the rest
   of the study.
3. **Widened-augmentation variant, built on `baseline` (Dice+CE), not the
   best-performing condition.** Chosen deliberately for a clean A/B:
   `baseline-500` vs. `baseline-500` + wider brightness/contrast/gamma (+
   possibly elastic deformation, currently disabled) isolates augmentation's
   effect on the `test_id`/`test_ood` gap without confounding it with a loss
   or sampling choice — the most direct next step on the anomaly below.

**Must exist *before* tomorrow night's run lands, so results→report has zero
engineering gap** (build in parallel, doesn't need GPU):

4. **Prediction ensembling in `evaluation/`.** Softmax-averaging across the
   final set of checkpoints — ready to apply the moment tomorrow's run
   finishes, not something to build afterward under presentation pressure.
5. **Per-site `test_ood` breakdown in `aggregate_results.py`/
   `plot_results.py`.** CLAUDE.md (2026-08-17): `test_ood` Dice ≥ `test_id`
   Dice in *every* condition so far — the opposite of ISLES'26's motivating
   premise — flagged as "worth investigating per-site... before writing this
   up." Currently only stratified by `size_bin` and pooled `split`. Needs to
   exist before tomorrow's results are the ones going in the report.
6. **Single-fold limitation stays a stated caveat, not a fix.** Already
   flagged under "Remaining risks" above — write it into the final report
   explicitly, since there's no follow-up run left to caveat it in later.
