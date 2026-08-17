# Execution plan

See `CLAUDE.md` for the running decisions log (why things deviate from
nnU-Net/project defaults) and working agreement. Update both when a plan step
here changes as a result of a new decision.

## Phase 0 — Verify the environment

```bash
python isles26.py init --raw-root "/path/to/ATLAS_R2.1_raw"
python isles26.py doctor --create-dirs --require-raw
python isles26.py prepare --dry-run
```

Inspect several discovered image/mask pairs manually before writing the dataset.

## Phase 0.5 — Split the data (train/val/test_id/test_ood)

```bash
python data_prep/split_dataset.py --raw-root "/path/to/ATLAS_R3.0_raw" --out-dir workspace/splits
```

Replaces nnU-Net's default unstratified random 5-fold CV with a single fixed
split, stratified by lesion-size bin, plus an out-of-distribution test set
made of entire held-out sites — see `CLAUDE.md` decisions log for why. Do not
proceed to `prepare` until `broken_cases.csv` from this step is small/stable
(i.e. the raw-data upload/transfer has actually finished).

## Phase 1 — Build and smoke-test the baseline

```bash
python isles26.py prepare
python isles26.py preprocess
python isles26.py train debug
python isles26.py train baseline
```

The debug run is a pipeline check only. Use the same fold, configuration, and
250-epoch budget for every real comparison (see `CLAUDE.md`).

## Phase 2 — Loss study

```bash
python isles26.py train losses
```

This trains Dice-only, focal, Tversky, and focal-Tversky variants. Reuse the baseline as the Dice+CE condition.

## Phase 3 — Lesion-aware sampling

```bash
python isles26.py train sampling
```

Inspect `workspace/case_metadata.csv` before training, especially the smallest cases and the `sampling_weight` distribution.

## Phase 4 — Evaluation and reporting

For each model prediction folder:

```bash
python isles26.py evaluate --pred-dir "/path/to/validation" --experiment baseline
```

After all experiments:

```bash
python isles26.py aggregate
python isles26.py plot
```

Report overall Dice/HD95 and results stratified by lesion-size bin. Include empty-prediction counts and qualitative failures.

## Future work (optional, not scheduled)

- **Widen intensity augmentation for cross-center generalization.** Try a
  trainer subclass with wider brightness/contrast/gamma ranges than nnU-Net's
  defaults; compare `test_id` vs. `test_ood` Dice gap against baseline before
  and after. Report `test_ood` per-center (site counts are unbalanced — see
  `CLAUDE.md` "Future considerations") rather than only pooled.
- **Domain-adversarial training** (gradient-reversal domain classifier on
  encoder features) as a stretch experiment, only if the gap above is still
  meaningful after the augmentation change — see `CLAUDE.md` for the
  reasoning against doing this first.

## Resource fallback

If GPU time is limited, reduce the number of folds but keep the exact same fold for every method. Do not treat the debug trainer as an experimental result. Avoid changing configuration, fold, loss, and sampling strategy simultaneously because the comparison becomes uninterpretable.
