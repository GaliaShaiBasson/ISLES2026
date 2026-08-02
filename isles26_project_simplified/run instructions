python -m pip install -r requirements.txt

python isles26.py init --raw-root "/path/to/ATLAS_R2.1_raw"
python isles26.py doctor --create-dirs --require-raw
python isles26.py prepare --dry-run
python isles26.py prepare
python isles26.py preprocess
python isles26.py train debug
python isles26.py train baseline



# ISLES26 Project Command Guide

## 1. General command format

Windows PowerShell:

```powershell
python .\isles26.py COMMAND [OPTIONS]
```

Linux:

```bash
python3 ./isles26.py COMMAND [OPTIONS]
```

You can display the complete command list with:

```powershell
python .\isles26.py --help
```

You can display the options for a specific command with:

```powershell
python .\isles26.py prepare --help
python .\isles26.py train --help
```

The available top-level commands are:

```text
init
doctor
prepare
preprocess
train
evaluate
aggregate
plot
```

There is currently no `predict` command in `isles26.py`. Prediction must be run with the nnU-Net prediction command separately.

---

# 2. Recommended workflow

A normal workflow is:

```powershell
python .\isles26.py init --raw-root "C:\path\to\ATLAS"
python .\isles26.py doctor --create-dirs --require-raw
python .\isles26.py prepare --dry-run
python .\isles26.py prepare
python .\isles26.py preprocess
python .\isles26.py train debug
python .\isles26.py train baseline
```

For your five-case debug dataset using dataset ID 2:

```powershell
python .\isles26.py init `
  --raw-root "C:\Users\Tevel Katzir\Downloads\project_debug_data" `
  --force

python .\isles26.py doctor --create-dirs --require-raw

python .\isles26.py prepare --dataset-id 2 --dry-run

python .\isles26.py prepare --dataset-id 2

python .\isles26.py preprocess --dataset-id 2

python .\isles26.py train debug --dataset-id 2
```

---

# 3. `init`

## Purpose

Creates the project `.env` configuration file and the local workspace directories.

Basic usage:

```powershell
python .\isles26.py init
```

## `--raw-root`

Sets the location of the original ATLAS dataset.

```powershell
python .\isles26.py init `
  --raw-root "C:\Users\Tevel Katzir\Downloads\project_debug_data"
```

The path is saved in:

```text
.env
```

under:

```text
ISLES26_RAW_ROOT
```

## `--force`

Replaces an existing `.env` file.

```powershell
python .\isles26.py init `
  --raw-root "C:\new\ATLAS\path" `
  --force
```

Without `--force`, an existing `.env` is retained.

Important: `--force` recreates the entire `.env` using the project defaults. Any manual changes in the old `.env` are replaced.

## Examples

Create initial configuration:

```powershell
python .\isles26.py init `
  --raw-root "D:\Datasets\ATLAS_R2.1"
```

Replace existing configuration:

```powershell
python .\isles26.py init `
  --raw-root "D:\Datasets\ATLAS_new" `
  --force
```

---

# 4. `doctor`

## Purpose

Checks whether the project environment is ready.

It checks:

* Python version
* nnU-Net directories
* nnU-Net executables
* installed `nnunetv2` version
* installed PyTorch version
* raw dataset path
* custom trainer discovery

Basic command:

```powershell
python .\isles26.py doctor
```

## `--create-dirs`

Creates missing nnU-Net storage directories.

```powershell
python .\isles26.py doctor --create-dirs
```

The directories normally include:

```text
workspace\nnUNet_raw
workspace\nnUNet_preprocessed
workspace\nnUNet_results
```

## `--require-raw`

Makes an invalid or missing raw dataset directory a blocking error.

```powershell
python .\isles26.py doctor --require-raw
```

Without this option, the raw path is still displayed, but it does not necessarily cause the command to fail.

## Recommended check

```powershell
python .\isles26.py doctor --create-dirs --require-raw
```

All lines should normally say `[OK]`.

---

# 5. `prepare`

## Purpose

Converts the original ATLAS dataset into nnU-Net format.

It creates a directory such as:

```text
workspace\nnUNet_raw\Dataset001_ATLAS
```

The generated dataset contains:

```text
imagesTr
labelsTr
dataset.json
```

It also creates:

```text
workspace\case_metadata.csv
```

## Basic command

```powershell
python .\isles26.py prepare
```

This uses the dataset root and defaults saved in `.env`.

---

## `--raw-root`

Overrides the raw dataset path for this command only.

```powershell
python .\isles26.py prepare `
  --raw-root "D:\Datasets\ATLAS_R2.1"
```

This does not permanently update `.env`.

Use it when testing a different source folder without changing configuration.

---

## `--dataset-id`

Sets the numeric nnU-Net dataset ID.

```powershell
python .\isles26.py prepare --dataset-id 2
```

This creates:

```text
Dataset002_ATLAS
```

The ID is formatted as three digits:

```text
1   -> Dataset001
2   -> Dataset002
25  -> Dataset025
```

The option must be typed exactly as:

```text
--dataset-id
```

Not:

```text
--dataset -id
```

---

## `--dataset-name`

Changes the text portion of the generated dataset folder.

```powershell
python .\isles26.py prepare `
  --dataset-id 2 `
  --dataset-name ATLAS_DEBUG
```

This creates:

```text
Dataset002_ATLAS_DEBUG
```

Avoid spaces and unusual symbols in the name.

---

## `--dry-run`

Scans the raw dataset without copying files.

```powershell
python .\isles26.py prepare --dry-run
```

It reports the total number of usable cases:

```text
Found 655 usable cases
```

It then prints only the first 10 cases as a preview.

The 10-case display is intentional. It does not mean only 10 cases will be processed during a real run.

For your debug dataset:

```powershell
python .\isles26.py prepare --dataset-id 2 --dry-run
```

Expected output:

```text
Found 5 usable cases
```

---

## `--overwrite`

Deletes and recreates an existing generated nnU-Net dataset folder.

```powershell
python .\isles26.py prepare `
  --dataset-id 2 `
  --overwrite
```

Use this when:

* the raw data changed;
* the wrong source directory was used;
* dataset preparation was interrupted;
* you want to rebuild the same dataset ID.

This affects the generated folder under `workspace\nnUNet_raw`. It does not delete the original ATLAS dataset.

Example:

```powershell
python .\isles26.py prepare `
  --dataset-id 2 `
  --dataset-name ATLAS `
  --overwrite
```

---

## `--print-only`

Prints the underlying command without executing it.

```powershell
python .\isles26.py prepare --print-only
```

This is different from `--dry-run`.

`--print-only`:

* does not scan the dataset;
* does not run the converter;
* only prints the command that would be launched.

`--dry-run`:

* actually launches the converter;
* scans the source directory;
* reports discovered cases;
* does not copy the data.

You can combine them, but doing so only prints the intended dry-run command:

```powershell
python .\isles26.py prepare --dry-run --print-only
```

---

# 6. `preprocess`

## Purpose

Runs nnU-Net planning and preprocessing.

It executes approximately:

```text
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Basic command:

```powershell
python .\isles26.py preprocess
```

The default dataset ID comes from `.env`, normally `1`.

## `--dataset-id`

Selects the dataset to preprocess.

```powershell
python .\isles26.py preprocess --dataset-id 2
```

For `Dataset002_ATLAS`, use ID `2`.

---

## `--no-verify`

Skips nnU-Net dataset-integrity verification.

```powershell
python .\isles26.py preprocess `
  --dataset-id 2 `
  --no-verify
```

Normally, do not use this option. Dataset verification catches missing files, incorrect names, geometry problems, and invalid labels.

Recommended:

```powershell
python .\isles26.py preprocess --dataset-id 2
```

Use `--no-verify` only when you understand why verification is failing and have independently checked the dataset.

---

## `--print-only`

Prints the nnU-Net preprocessing command without running it.

```powershell
python .\isles26.py preprocess `
  --dataset-id 2 `
  --print-only
```

---

# 7. `train`

## Purpose

Runs one trainer or a predefined group of trainers.

General syntax:

```powershell
python .\isles26.py train EXPERIMENT [OPTIONS]
```

The experiment name is required.

Available experiment names:

```text
baseline
debug
dice
focal
tversky
focal-tversky
losses
sampling
```

---

# 8. Training experiment types

## `debug`

Runs the shortened debug trainer:

```text
nnUNetTrainerDebugFast
```

Default debug settings:

```text
configuration: 2d
device: cpu
```

Command:

```powershell
python .\isles26.py train debug --dataset-id 2
```

This is intended to verify that:

* the dataset is readable;
* preprocessing succeeded;
* the custom trainer is discoverable;
* training can start;
* checkpoints and result folders can be created.

It is not intended to produce a final model.

You can run debug on a GPU:

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --device cuda
```

---

## `baseline`

Runs the standard nnU-Net trainer:

```text
nnUNetTrainer
```

Command:

```powershell
python .\isles26.py train baseline --dataset-id 2
```

For non-debug experiments, defaults normally are:

```text
configuration: 3d_fullres
device: cuda
fold: 0
GPU count: 1
```

---

## `dice`

Runs:

```text
nnUNetTrainerDiceOnly
```

Command:

```powershell
python .\isles26.py train dice --dataset-id 2
```

This tests the custom Dice-only loss.

---

## `focal`

Runs:

```text
nnUNetTrainerFocal
```

Command:

```powershell
python .\isles26.py train focal --dataset-id 2
```

This tests focal loss, which emphasizes harder examples.

---

## `tversky`

Runs:

```text
nnUNetTrainerTversky
```

Command:

```powershell
python .\isles26.py train tversky --dataset-id 2
```

This tests Tversky loss, which can weight false positives and false negatives differently.

---

## `focal-tversky`

Runs:

```text
nnUNetTrainerFocalTversky
```

Command:

```powershell
python .\isles26.py train focal-tversky --dataset-id 2
```

---

## `losses`

Runs all four custom loss experiments sequentially:

```text
nnUNetTrainerDiceOnly
nnUNetTrainerFocal
nnUNetTrainerTversky
nnUNetTrainerFocalTversky
```

Command:

```powershell
python .\isles26.py train losses --dataset-id 2
```

The trainers run one after another, not simultaneously.

This may take a long time on the full dataset.

---

## `sampling`

Runs:

```text
nnUNetTrainerLesionAwareSampling
```

Command:

```powershell
python .\isles26.py train sampling --dataset-id 2
```

This trainer requires:

```text
workspace\case_metadata.csv
```

That file is created by `prepare`.

If it is missing, the command stops and asks you to run dataset preparation first.

---

# 9. Training options

## `--trainer`

Overrides the trainer associated with the selected experiment.

Example:

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --trainer nnUNetTrainerDebugFast
```

Here, `baseline` is only being used to satisfy the required experiment argument. The specified trainer takes precedence.

This option is mainly for advanced testing.

---

## `--dataset-id`

Selects the nnU-Net dataset.

```powershell
python .\isles26.py train baseline --dataset-id 2
```

Use the same ID passed to `prepare` and `preprocess`.

---

## `--fold`

Selects the cross-validation fold.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --fold 0
```

Typical folds are:

```text
0
1
2
3
4
```

You can also pass:

```text
all
```

only when supported by the underlying nnU-Net command and workflow.

Example:

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --fold 1
```

For a tiny five-case debug dataset, fold behavior may not be statistically meaningful.

---

## `--configuration`

Overrides the nnU-Net configuration.

Common values include:

```text
2d
3d_fullres
3d_lowres
3d_cascade_fullres
```

Not every dataset will have every configuration.

Example:

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --configuration 2d
```

The normal default is:

```text
3d_fullres
```

The `debug` experiment defaults to:

```text
2d
```

---

## `--device`

Selects the computing device.

Allowed values:

```text
cuda
cpu
mps
```

Windows with an NVIDIA GPU:

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --device cuda
```

CPU:

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --device cpu
```

`mps` is for Apple Silicon/macOS, not Windows.

---

## `--num-gpus`

Sets the number of GPUs.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --num-gpus 2
```

The default is one GPU.

Do not set this higher than the number of available compatible GPUs.

---

## `--preprocess`

Runs preprocessing immediately before training.

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --preprocess
```

This is equivalent to running:

```powershell
python .\isles26.py preprocess --dataset-id 2
python .\isles26.py train debug --dataset-id 2
```

This is convenient for a first run.

---

## `--continue`

Continues an interrupted training run from its latest checkpoint.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --continue
```

Internally, this passes nnU-Net’s `--c` option.

Use it when:

* training was interrupted;
* the computer restarted;
* you intentionally stopped and want to resume.

A compatible checkpoint must already exist.

---

## `--validate-only`

Runs validation without training.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --validate-only
```

Internally, this passes:

```text
--val
```

Use this after a checkpoint has already been trained.

---

## `--val-best`

Uses the best checkpoint for validation instead of the final checkpoint.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --validate-only `
  --val-best
```

This is typically combined with `--validate-only`.

---

## `--npz`

Saves softmax probability outputs as compressed `.npz` files during validation.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --validate-only `
  --npz
```

These files are useful for ensembling but consume additional disk space.

---

## `--disable-checkpointing`

Disables normal checkpoint saving.

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --disable-checkpointing
```

This can reduce storage use during temporary tests.

Do not use it for a long experiment that you may need to resume.

---

## `--print-only`

Prints the training command without starting training.

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --print-only
```

This is useful for checking:

* dataset ID;
* fold;
* trainer;
* configuration;
* device;
* GPU count.

Example output:

```text
nnUNetv2_train 2 3d_fullres 0
```

---

# 10. Common training examples

## Debug training on CPU

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --device cpu
```

## Debug training with preprocessing

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --preprocess
```

## Standard baseline training

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --configuration 3d_fullres `
  --fold 0 `
  --device cuda
```

## Continue baseline training

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --continue
```

## Validate the best checkpoint

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --validate-only `
  --val-best
```

## Run all custom loss experiments

```powershell
python .\isles26.py train losses `
  --dataset-id 2 `
  --device cuda
```

## Run lesion-aware sampling

```powershell
python .\isles26.py train sampling `
  --dataset-id 2 `
  --device cuda
```

---

# 11. Prediction

The unified runner does not currently contain a `predict` command.

Prediction must be run using `nnUNetv2_predict`.

General format:

```powershell
nnUNetv2_predict `
  -i "PATH_TO_INPUT_IMAGES" `
  -o "PATH_TO_PREDICTIONS" `
  -d DATASET_ID `
  -c CONFIGURATION `
  -f FOLD `
  -tr TRAINER_NAME
```

For example:

```powershell
nnUNetv2_predict `
  -i "C:\data\test_images" `
  -o "C:\data\predictions\baseline" `
  -d 2 `
  -c 3d_fullres `
  -f 0
```

For a custom trainer:

```powershell
nnUNetv2_predict `
  -i "C:\data\test_images" `
  -o "C:\data\predictions\focal" `
  -d 2 `
  -c 3d_fullres `
  -f 0 `
  -tr nnUNetTrainerFocal
```

Input image names must follow nnU-Net naming conventions, such as:

```text
CASE001_0000.nii.gz
CASE002_0000.nii.gz
```

---

# 12. `evaluate`

## Purpose

Computes per-case segmentation metrics for a prediction directory.

The metrics include:

```text
Dice
HD95
```

Required options:

```text
--pred-dir
--experiment
```

Basic example:

```powershell
python .\isles26.py evaluate `
  --pred-dir "C:\data\predictions\baseline" `
  --experiment "baseline"
```

By default, ground-truth labels are read from:

```text
workspace\nnUNet_raw\DatasetXXX_NAME\labelsTr
```

The default dataset folder is determined by the dataset ID and name in `.env`.

---

## `--pred-dir`

Path containing predicted `.nii.gz` masks.

```powershell
--pred-dir "C:\data\predictions\baseline"
```

The prediction filenames must correspond to the ground-truth case names.

---

## `--experiment`

Label written into the results CSV.

```powershell
--experiment "baseline"
```

Examples:

```text
baseline
dice_only
focal
tversky
lesion_aware_sampling
```

This value is also used to construct the default output filename.

For example:

```text
baseline
```

produces approximately:

```text
workspace\evaluation\results_baseline.csv
```

---

## `--gt-dir`

Overrides the ground-truth directory.

```powershell
python .\isles26.py evaluate `
  --pred-dir "C:\data\predictions\baseline" `
  --experiment "baseline" `
  --gt-dir "C:\data\ground_truth"
```

Use this when ground truth is not in the generated nnU-Net `labelsTr` directory.

---

## `--out-csv`

Sets a custom result CSV path.

```powershell
python .\isles26.py evaluate `
  --pred-dir "C:\data\predictions\baseline" `
  --experiment "baseline" `
  --out-csv "C:\data\results\baseline_metrics.csv"
```

---

## `--print-only`

Prints the evaluation command without running it.

```powershell
python .\isles26.py evaluate `
  --pred-dir "C:\data\predictions\baseline" `
  --experiment "baseline" `
  --print-only
```

---

# 13. `aggregate`

## Purpose

Combines result CSV files from multiple experiments.

Basic command:

```powershell
python .\isles26.py aggregate
```

When no files are supplied, it searches:

```text
workspace\evaluation
```

for files matching:

```text
results_*.csv
```

It ignores the already-combined file:

```text
results.csv
```

It creates:

```text
workspace\evaluation\results.csv
workspace\evaluation\summary_by_experiment.csv
workspace\evaluation\summary_by_size_bin.csv
```

---

## Supplying specific files

```powershell
python .\isles26.py aggregate `
  "workspace\evaluation\results_baseline.csv" `
  "workspace\evaluation\results_focal.csv"
```

All supplied CSV files are combined.

---

## `--print-only`

Prints the aggregation command without running it.

```powershell
python .\isles26.py aggregate --print-only
```

Or with selected files:

```powershell
python .\isles26.py aggregate `
  "workspace\evaluation\results_baseline.csv" `
  "workspace\evaluation\results_focal.csv" `
  --print-only
```

---

# 14. `plot`

## Purpose

Creates figures from the aggregated results CSV.

Basic command:

```powershell
python .\isles26.py plot
```

By default it reads:

```text
workspace\evaluation\results.csv
```

and writes figures to:

```text
workspace\figures
```

Run `aggregate` before `plot`.

---

## `--results-csv`

Uses a different combined results file.

```powershell
python .\isles26.py plot `
  --results-csv "C:\data\results\results.csv"
```

---

## `--out-dir`

Changes the figure output directory.

```powershell
python .\isles26.py plot `
  --out-dir "C:\data\figures"
```

---

## `--print-only`

Prints the plotting command without generating figures.

```powershell
python .\isles26.py plot --print-only
```

---

# 15. Important configuration values in `.env`

The `.env` file stores defaults used when command-line options are omitted.

Typical contents:

```text
ISLES26_RAW_ROOT="C:\path\to\ATLAS"
ISLES26_WORKSPACE="workspace"
ISLES26_CASE_METADATA_CSV="workspace/case_metadata.csv"
ISLES26_RESULTS_DIR="workspace/evaluation"
ISLES26_FIGURES_DIR="workspace/figures"
ISLES26_DATASET_ID="1"
ISLES26_DATASET_NAME="ATLAS"
ISLES26_CONFIGURATION="3d_fullres"
ISLES26_FOLD="0"
ISLES26_DEVICE="cuda"
ISLES26_NUM_GPUS="1"
nnUNet_raw="workspace/nnUNet_raw"
nnUNet_preprocessed="workspace/nnUNet_preprocessed"
nnUNet_results="workspace/nnUNet_results"
```

Command-line options override these values for the current command.

For example:

```powershell
python .\isles26.py prepare --dataset-id 2
```

uses ID 2 for that run even if `.env` says ID 1.

However, it does not permanently change `.env`.

---

# 16. Important dataset-ID consistency rule

Use the same dataset ID through preparation, preprocessing, training, prediction, and evaluation.

For example:

```powershell
python .\isles26.py prepare --dataset-id 2
python .\isles26.py preprocess --dataset-id 2
python .\isles26.py train debug --dataset-id 2
python .\isles26.py train baseline --dataset-id 2
```

A frequent mistake is:

```powershell
python .\isles26.py prepare --dataset-id 2
python .\isles26.py preprocess
```

The second command uses the `.env` default, which may still be dataset ID 1.

Until `.env` is changed, explicitly pass:

```text
--dataset-id 2
```

to each relevant command.

---

# 17. Complete debug-data workflow

For your current five-case folder:

```powershell
python .\isles26.py init `
  --raw-root "C:\Users\Tevel Katzir\Downloads\project_debug_data" `
  --force

python .\isles26.py doctor `
  --create-dirs `
  --require-raw

python .\isles26.py prepare `
  --dataset-id 2 `
  --dry-run

python .\isles26.py prepare `
  --dataset-id 2 `
  --overwrite

python .\isles26.py preprocess `
  --dataset-id 2

python .\isles26.py train debug `
  --dataset-id 2 `
  --device cpu
```

After the debug run works, test GPU training:

```powershell
python .\isles26.py train debug `
  --dataset-id 2 `
  --device cuda
```

Then run the baseline:

```powershell
python .\isles26.py train baseline `
  --dataset-id 2 `
  --configuration 3d_fullres `
  --fold 0 `
  --device cuda
```

---

# 18. Complete full-dataset workflow

```powershell
python .\isles26.py init `
  --raw-root "D:\Datasets\ATLAS_R2.1" `
  --force

python .\isles26.py doctor `
  --create-dirs `
  --require-raw

python .\isles26.py prepare `
  --dataset-id 1 `
  --dry-run

python .\isles26.py prepare `
  --dataset-id 1 `
  --overwrite

python .\isles26.py preprocess `
  --dataset-id 1

python .\isles26.py train debug `
  --dataset-id 1

python .\isles26.py train baseline `
  --dataset-id 1

python .\isles26.py train losses `
  --dataset-id 1

python .\isles26.py train sampling `
  --dataset-id 1
```

---

# 19. Troubleshooting commands

Check configuration and dependencies:

```powershell
python .\isles26.py doctor --require-raw
```

Check which raw cases are discovered:

```powershell
python .\isles26.py prepare --dry-run
```

Check the generated command without running it:

```powershell
python .\isles26.py prepare --print-only
python .\isles26.py preprocess --dataset-id 2 --print-only
python .\isles26.py train debug --dataset-id 2 --print-only
```

Show command-specific help:

```powershell
python .\isles26.py init --help
python .\isles26.py doctor --help
python .\isles26.py prepare --help
python .\isles26.py preprocess --help
python .\isles26.py train --help
python .\isles26.py evaluate --help
python .\isles26.py aggregate --help
python .\isles26.py plot --help
```

---

# 20. Most important practical notes

1. Always run commands from the folder containing `isles26.py`.

2. Use `--dataset-id`, with no space inside the option name.

3. `--dry-run` scans the data but does not copy it.

4. `--print-only` only prints a command and performs no scan or processing.

5. Use `--overwrite` when rebuilding an existing prepared dataset.

6. Use the same dataset ID for preparation, preprocessing, and training.

7. The runner does not currently provide a prediction subcommand.

8. Run `debug` before starting long experiments.

9. Do not use `--disable-checkpointing` for important long-running training.

10. The `losses` experiment runs four full training experiments sequentially.
