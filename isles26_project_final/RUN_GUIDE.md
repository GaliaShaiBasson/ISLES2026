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

## `predict`

```text
python isles26.py predict EXPERIMENT --input-dir PATH
```

Experiments: `baseline`, `focal`, `tversky`, `sampling`, `curriculum`.

Options include `--out-dir`, `--dataset-id`, `--configuration`, `--folds`, `--checkpoint`, `--device`, `--save-probabilities`, `--continue-prediction`, `--disable-tta`, and `--print-only`.

## `evaluate`

```text
python isles26.py evaluate --experiment NAME [--pred-dir PATH]
```

Computes per-case Dice and HD95 and joins the prepared metadata. If `--pred-dir` is omitted, the runner uses the default prediction folder for the named experiment.

Use `--dataset-id`/`--dataset-name` when evaluating a non-default dataset.

## `aggregate`

```text
python isles26.py aggregate
```

Combines `workspace/evaluation/results_*.csv` and writes summaries by experiment, lesion size, center, and center×size.

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
