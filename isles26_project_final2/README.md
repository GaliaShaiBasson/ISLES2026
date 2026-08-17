# ATLAS lesion-size-aware sampling experiments with nnU-Net v2

This project provides one cross-platform runner for a reproducible five-fold
comparison of lesion-size-aware sampling methods:

```bash
python isles26.py --help
```

It targets the **ATLAS R2.1** folder layout and nnU-Net **2.8.1**. The project contains five scientific experiments plus a debug smoke test:

1. `baseline` — standard nnU-Net trainer
2. `focal` — Focal loss
3. `tversky` — Tversky loss
4. `sampling` — fixed inverse-lesion-volume case sampling
5. `curriculum` — uniform → mild → full inverse-volume sampling over training

There is intentionally **no size-balanced sampling experiment**.

## Setup

Create a virtual environment, install the hardware-appropriate PyTorch build, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

Create/update the local project configuration:

```bash
python isles26.py init --raw-root "/path/to/ATLAS_R2.1" --dataset-id 1
python isles26.py doctor --create-dirs --require-raw
```

On Windows PowerShell the same runner is used:

```powershell
python .\isles26.py init --raw-root "D:\Datasets\ATLAS_R2.1" --dataset-id 1
python .\isles26.py doctor --create-dirs --require-raw
```

Running `init` again updates only the settings you specify. `--force` resets all settings to defaults first. Project `.env` values take precedence over inherited shell variables, avoiding stale path overrides.

## Prepare and preprocess

Inspect discovery first:

```bash
python isles26.py prepare --dry-run
```

The dry run reports the full number of usable cases and prints at most 10 examples. A real run processes all discovered cases.

Create the nnU-Net dataset and metadata, then preprocess:

```bash
python isles26.py prepare
python isles26.py preprocess
python isles26.py make-splits
```

`make-splits` writes a deterministic five-fold `splits_final.json`. Sessions
from the same subject stay together, while fold assignment is balanced across
acquisition centers and lesion-size bins. Create the splits after preprocessing
and before the first scientific training run. A fold-composition CSV is written
to `workspace/metadata/` for the report.

Metadata is stored per dataset, for example:

```text
workspace/metadata/Dataset001_ATLAS_case_metadata.csv
```

It contains `case_id`, lesion volume, lesion-size bin, fixed sampling weight, acquisition center, and any readable one-row ATLAS metadata fields.

## Smoke test

Before full GPU training:

```bash
python isles26.py train debug
```

The debug trainer uses a short schedule and is not for reported results.

## Train experiments

Run experiments separately:

```bash
python isles26.py train baseline
python isles26.py train focal
python isles26.py train tversky
python isles26.py train sampling
python isles26.py train curriculum
```

Or run all five sequentially:

```bash
python isles26.py train all
```

For final results, train every method on the same five folds:

```bash
python isles26.py train all --folds 0 1 2 3 4
```

Each non-dry training launch writes a JSON run manifest containing the command,
environment, package versions, configuration, and source hashes under
`workspace/run_manifests/`.

Useful options:

```bash
python isles26.py train baseline --fold 0 --configuration 3d_fullres
python isles26.py train baseline --continue
python isles26.py train sampling --print-only
python isles26.py train curriculum --device cuda
```

Model checkpoints are written by nnU-Net under:

```text
workspace/nnUNet_results/DatasetXXX_NAME/<Trainer>__nnUNetPlans__<configuration>/fold_X/
```

Typical checkpoint files are `checkpoint_best.pth`, `checkpoint_latest.pth` during training, and `checkpoint_final.pth` at completion.

## Sampling methods

### Fixed lesion-size-aware sampling

The `sampling` experiment samples training cases with probability proportional to:

```text
1 / lesion_volume_mm3
```

Small-lesion cases are therefore drawn more often. Validation remains uniformly sampled.

### Curriculum lesion-size-aware sampling

The `curriculum` experiment changes the **continuous** inverse-volume power over the run:

```text
first third   : volume^0     -> uniform case sampling
middle third  : volume^-0.5  -> mild small-lesion emphasis
final third   : volume^-1    -> full fixed lesion-size-aware sampling
```

At the two phase boundaries, the data loaders are rebuilt so the new probabilities reach multiprocessing workers on Linux and Windows.

## Prediction

Inputs must use nnU-Net channel naming, for example `case_0000.nii.gz`.

```bash
python isles26.py predict baseline --input-dir "/path/to/imagesTs"
python isles26.py predict sampling --input-dir "/path/to/imagesTs"
python isles26.py predict curriculum --input-dir "/path/to/imagesTs"
```

Predictions default to:

```text
workspace/predictions/DatasetXXX_NAME/<experiment>/
```

## Evaluation

Evaluate one prediction folder:

```bash
python isles26.py evaluate --experiment baseline
python isles26.py evaluate --experiment focal
python isles26.py evaluate --experiment tversky
python isles26.py evaluate --experiment sampling
python isles26.py evaluate --experiment curriculum
```

For scientific results, evaluate the out-of-fold validation outputs from all
five folds:

```bash
python isles26.py evaluate --experiment baseline --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment focal --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment tversky --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment sampling --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment curriculum --from-training-validation --folds 0 1 2 3 4
```

Evaluation is strict by default: every case expected by the selected fold must
have a prediction and ground truth. Each CSV contains Dice, raw HD95, a
penalized HD95, empty-prediction indicators, lesion volume, size bin, center,
fold, and attached metadata. If exactly one mask is empty, raw HD95 is undefined
and `hd95_penalized_mm` uses the physical image diagonal. The raw value and the
penalty flag remain in the CSV so this choice is auditable.

## Analysis

One command aggregates all experiment CSVs and generates figures:

```bash
python isles26.py analyze
```

The analysis produces:

```text
workspace/evaluation/results.csv
workspace/evaluation/summary_by_experiment.csv
workspace/evaluation/summary_by_size_bin.csv
workspace/evaluation/summary_by_center.csv
workspace/evaluation/summary_by_size_and_center.csv
workspace/evaluation/paired_comparisons.csv

workspace/figures/overall_dice.png
workspace/figures/overall_hd95.png
workspace/figures/empty_prediction_rate.png
workspace/figures/dice_by_size_bin.png
workspace/figures/dice_by_center.png
workspace/figures/dice_vs_volume_scatter.png
workspace/figures/dice_center_by_size_<experiment>.png
```

Aggregation rejects unequal experiment cohorts. `paired_comparisons.csv`
contains case-paired mean improvements, 95% bootstrap confidence intervals,
Wilcoxon signed-rank p-values, and Holm corrections. Positive improvement means
better for both metrics; the HD95 difference is therefore baseline minus
candidate. Center and center×size outputs are descriptive robustness analyses
and always include sample counts.

## Recommended full workflow

```bash
python isles26.py init --raw-root "/path/to/ATLAS_R2.1" --dataset-id 1
python isles26.py doctor --create-dirs --require-raw
python isles26.py prepare --dry-run
python isles26.py prepare
python isles26.py preprocess
python isles26.py make-splits
python isles26.py train debug
python isles26.py train all --folds 0 1 2 3 4
```

After predictions or fold validation outputs are available:

```bash
python isles26.py evaluate --experiment baseline --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment focal --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment tversky --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment sampling --from-training-validation --folds 0 1 2 3 4
python isles26.py evaluate --experiment curriculum --from-training-validation --folds 0 1 2 3 4
python isles26.py analyze
```

## Verification

Run the lightweight tests with:

```bash
python -m unittest discover -s tests -v
```

The PyTorch loss test requires the hardware-appropriate PyTorch build. A real
scientific run additionally requires ATLAS data and nnU-Net; passing unit tests
does not substitute for the debug trainer smoke test or full cross-validation.

## Results and report

This source archive intentionally does not claim experimental results. Populate
the generated CSVs/figures only from completed runs, then follow
`REPORT_CHECKLIST.md` and `EXPERIMENT_PROTOCOL.md`. Do not write conclusions
before checking cohort completeness, failure rates, paired confidence intervals,
and subgroup sample counts.

## Required citations and upstream code

- S.-L. Liew et al., “A large, curated, open-source stroke neuroimaging dataset
  to improve lesion segmentation algorithms,” *Scientific Data* (2022):
  https://doi.org/10.1038/s41597-022-01401-7
- F. Isensee et al., “nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation,” *Nature Methods* 18, 203–211 (2021):
  https://doi.org/10.1038/s41592-020-01008-z
- nnU-Net source: https://github.com/MIC-DKFZ/nnUNet
- ATLAS challenge/data page: https://atlas.grand-challenge.org/

See `RUN_GUIDE.md` for command-by-command details.
