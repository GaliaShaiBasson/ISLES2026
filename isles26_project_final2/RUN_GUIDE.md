# Command guide

All commands work from Windows PowerShell and Linux/macOS terminals. Run them from the directory containing `isles26.py`.

## `init`

Create or update `.env`:

```text
python isles26.py init --raw-root PATH --dataset-id ID
```

Optional settings: `--dataset-name`, `--configuration`, `--fold`, `--device`.

`--force` resets all settings to project defaults before applying the supplied options. Without `--force`, only supplied settings are updated.

## `doctor`

```text
python isles26.py doctor --create-dirs --require-raw
```

Checks Python, nnU-Net commands, nnU-Net version, PyTorch compatibility, raw-root existence, storage directories, and custom trainer discovery.

## `prepare`

```text
python isles26.py prepare [--dataset-id ID] [--dry-run] [--overwrite]
```

- `--dry-run`: discover cases but do not copy them; prints at most 10 examples.
- `--overwrite`: replace an existing generated `DatasetXXX_NAME` folder.
- `--raw-root`: use a different source folder for this run.
- `--dataset-id`: override the configured dataset ID for this run.
- `--dataset-name`: override the configured dataset name for this run.
- `--print-only`: print the converter command without running it.

## `preprocess`

```text
python isles26.py preprocess [--dataset-id ID]
```

Runs `nnUNetv2_plan_and_preprocess` with dataset integrity checking. `--no-verify` disables the integrity check. `--print-only` prints the command.

## `make-splits`

```text
python isles26.py make-splits [--dataset-id ID] [--seed 2026]
```

Creates `splits_final.json` in the preprocessed dataset directory. Assignment
is deterministic, keeps sessions from the same subject together, and balances
fold sizes while distributing acquisition-center × lesion-size strata. Run it
after preprocessing and before training. Use `--overwrite` only when you
intentionally want a new split definition; never change splits midway through
an experiment.

## `train`

```text
python isles26.py train EXPERIMENT
```

Experiments:

- `baseline`: standard `nnUNetTrainer`
- `focal`: `nnUNetTrainerFocal`
- `tversky`: `nnUNetTrainerTversky`
- `sampling`: fixed inverse-volume case sampler
- `curriculum`: uniform → inverse-sqrt-volume → inverse-volume sampler
- `all`: runs the five experiments above sequentially
- `debug`: short CPU/2D pipeline smoke test

Common options:

```text
--dataset-id ID
--dataset-name NAME
--fold FOLD
--folds 0 1 2 3 4
--configuration 3d_fullres
--device cuda|cpu|mps
--num-gpus N
--preprocess
--continue
--validate-only
--val-best
--npz
--disable-checkpointing
--print-only
```

`--fold` and `--folds` are mutually exclusive. The final comparison should use
the same five folds for every experiment:

```text
python isles26.py train all --folds 0 1 2 3 4
```

Every launched training writes an immutable run manifest under
`workspace/run_manifests/`.

## `predict`

```text
python isles26.py predict EXPERIMENT --input-dir PATH
```

Experiments: `baseline`, `focal`, `tversky`, `sampling`, `curriculum`.

Options include `--out-dir`, `--dataset-id`, `--configuration`, `--folds`, `--checkpoint`, `--device`, `--save-probabilities`, `--continue-prediction`, `--disable-tta`, and `--print-only`.

## `evaluate`

For the recommended five-fold out-of-fold evaluation:

```text
python isles26.py evaluate --experiment NAME --from-training-validation --folds 0 1 2 3 4
```

The runner resolves each trainer's `fold_X/validation` directory, verifies it
against the exact validation cases in `splits_final.json`, and writes one CSV
per fold. Missing expected predictions are fatal.

For a separate prediction folder:

```text
python isles26.py evaluate --experiment NAME --pred-dir PATH
python isles26.py evaluate --experiment NAME --pred-dir PATH --expected-cases-file CASES.csv
```

Without an explicit case list, every case in the metadata CSV is expected.
`--allow-missing` exists only for debugging and should never be used for a
reported table. Raw HD95 is retained, while `hd95_penalized_mm` uses the
physical image diagonal when exactly one mask is empty. Empty-prediction and
HD95-defined rates are always summarized.

## `aggregate`

```text
python isles26.py aggregate
```

Combines `workspace/evaluation/results_*.csv`, rejects unequal case sets across
experiments, and writes summaries by experiment, lesion size, center, and
center×size. It also writes `paired_comparisons.csv` with paired bootstrap
confidence intervals, Wilcoxon tests, and Holm-adjusted p-values.

## `plot`

```text
python isles26.py plot
```

Generates overall, lesion-size, center, center×size, and volume-scatter plots.

## `analyze`

```text
python isles26.py analyze
```

Runs `aggregate` and `plot` in sequence. This is the simplest final-analysis command.
