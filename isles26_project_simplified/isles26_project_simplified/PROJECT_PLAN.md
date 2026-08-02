# Execution plan

## Phase 0 — Verify the environment

```bash
python isles26.py init --raw-root "/path/to/ATLAS_R2.1_raw"
python isles26.py doctor --create-dirs --require-raw
python isles26.py prepare --dry-run
```

Inspect several discovered image/mask pairs manually before writing the dataset.

## Phase 1 — Build and smoke-test the baseline

```bash
python isles26.py prepare
python isles26.py preprocess
python isles26.py train debug
python isles26.py train baseline
```

The debug run is a pipeline check only. Use the same fold and configuration for every real comparison.

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

## Resource fallback

If GPU time is limited, reduce the number of folds but keep the exact same fold for every method. Do not treat the debug trainer as an experimental result. Avoid changing configuration, fold, loss, and sampling strategy simultaneously because the comparison becomes uninterpretable.
