# ISLES'26 Stroke Lesion Segmentation — Course Project

## Chosen plan

**Option 1: nnU-Net v2 baseline + one focused, well-motivated improvement.**

The single research thread running through this project is: *stroke lesions are
small relative to the brain volume, causing severe foreground/background
imbalance that hurts standard Dice/CE training.* We attack this from two
complementary, individually-cheap angles instead of a grab-bag of unrelated
tricks:

1. **Loss function study (A):** Dice vs Dice+CE vs Focal vs Tversky vs Focal
   Tversky — all designed with class imbalance in mind to varying degrees.
2. **Lesion-aware sampling (D):** oversample cases/patches with small lesions
   during training, on top of nnU-Net's default foreground oversampling.

Both are cheap to implement as nnU-Net v2 "trainer variants" (no architecture
changes, no extra models, no ensembling), and both are analyzed through the
same lens: **does Dice/HD95 improve specifically on small lesions**, without
regressing on large ones. That's the paper-shaped question.

## Project structure

```
isles26_project/
├── README.md                          <- this file
├── PROJECT_PLAN.md                    <- day-by-day plan mapped to your timeline
├── requirements.txt
├── data_prep/
│   ├── prepare_isles26_dataset.py     <- raw ISLES'26 -> nnU-Net raw format
│   └── metadata_utils.py              <- lesion volume, size bins, metadata join
├── custom_trainers/
│   ├── losses.py                      <- Focal / Tversky / Focal-Tversky losses
│   ├── nnUNetTrainerLossVariants.py   <- trainer subclasses for the loss study
│   ├── nnUNetTrainerLesionAwareSampling.py  <- trainer + dataloader for sampling study
│   └── install_custom_trainers.py     <- copies the above into your nnunetv2 install
├── training/
│   ├── run_baseline.sh
│   ├── run_loss_experiments.sh
│   └── run_sampling_experiment.sh
├── evaluation/
│   ├── compute_metrics.py             <- Dice + HD95 per case, stratified
│   └── aggregate_results.py           <- combines runs into summary tables
└── analysis/
    └── plot_results.py                <- figures for the report
```

## How the pieces connect

1. `data_prep/prepare_isles26_dataset.py` converts your raw ISLES'26 download
   into nnU-Net's expected `Dataset0XX_ISLES26` layout, and **also** writes
   `case_metadata.csv` with, per case: lesion volume (voxels/mm³), a
   small/medium/large size bin, and any provided metadata (center, chronicity,
   days-post-stroke). This CSV is the backbone of both the sampling trainer
   and the stratified evaluation later — build it once, use it everywhere.

2. `custom_trainers/install_custom_trainers.py` copies the loss and sampling
   trainer files into your installed `nnunetv2` package, following nnU-Net's
   own convention for custom trainers (a `-tr <ClassName>` flag at train time
   auto-discovers any `nnUNetTrainer` subclass placed under
   `nnunetv2/training/nnUNetTrainer/variants/`). This is the standard,
   supported way to extend nnU-Net without forking it.

3. `training/*.sh` are thin wrappers around the standard
   `nnUNetv2_plan_and_preprocess` / `nnUNetv2_train` CLI, just pointing at the
   right trainer classes and dataset ID.

4. `evaluation/compute_metrics.py` computes Dice and HD95 per case from
   nnU-Net's prediction folders, joins with `case_metadata.csv`, and writes
   one row per (case, experiment) to `results.csv`.

5. `analysis/plot_results.py` reads `results.csv` and produces the figures
   you actually want for a report: overall Dice/HD95 per method, Dice by
   lesion-size bin per method (the key plot), and per-center performance.

## What you need to supply

- **GPU + the actual ISLES'26 dataset.** I don't have access to either in
  this environment, so I can't run training or verify against the real data
  format. I've made the raw-data assumptions explicit and easy to adjust in
  `prepare_isles26_dataset.py` — check the `RAW_LAYOUT` docstring at the top
  and adjust the glob patterns to match what you actually download.
- **nnunetv2 installed** (`pip install nnunetv2`) in the environment you'll
  train in.
- Environment variables nnU-Net expects: `nnUNet_raw`, `nnUNet_preprocessed`,
  `nnUNet_results` (see `requirements.txt` / official nnU-Net docs for these —
  they're just folder paths).

## Windows notes

`training/*.sh` are bash (Linux/Mac). Matching `training/*.ps1` PowerShell
versions are provided for Windows — same flags, just `-DatasetId`/`-Fold`/
`-Config` instead of positional args, e.g.:

```powershell
.\run_baseline.ps1 -DatasetId 1 -Fold 0 -Config 2d
```

Two Windows-specific gotchas:
- `$env:VAR = "..."` only persists for the current PowerShell window. Re-set
  `nnUNet_raw`/`nnUNet_preprocessed`/`nnUNet_results` each new session, or
  set them permanently via System Properties → Environment Variables.
- nnU-Net's data-augmentation workers use `spawn` on Windows (vs `fork` on
  Linux/Mac), which starts slower and can occasionally misbehave on older
  nnU-Net versions. If training hangs or errors around multiprocessing,
  reduce workers first: `$env:nnUNet_n_proc_DA = "2"`.

## Concrete example: ATLAS R2.1 layout

`data_prep/prepare_isles26_dataset.py` is written for the actual ATLAS R2.1
download structure:

```
<raw_root>/Training_Raw/R0XX/sub-.../ses-.../anat/
    sub-..._metadata.csv
    sub-..._space-orig_desc-brain_T1w.nii.gz
    sub-..._space-orig_label-lesion_desc-T1lesion_mask.nii.gz
```

The `R0XX` folder is used as the `center` metadata field automatically, and
every session's own `metadata.csv` is read and merged in. On Windows, with
a path containing spaces, quote it:

```
python prepare_isles26_dataset.py --raw-root "C:\Users\Tevel Katzir\Downloads\ATLAS_R2.1_raw" --dry-run
```

Check the dry-run output lists your cases with the right center and
metadata file, then drop `--dry-run` (with `nnUNet_raw` set) to actually
write the nnU-Net dataset + `case_metadata.csv`.

## Running on CPU

Feasible for **pipeline debugging only**, not for real training. nnU-Net
defines an "epoch" as a fixed 250 iterations regardless of dataset size, so
subsetting your data doesn't shorten training — per-iteration cost (a 3D
patch through the network) does, and that's what's slow on CPU.

To smoke-test the whole pipeline in minutes:

```bash
nnUNetv2_train <DATASET_ID> 2d 0 -tr nnUNetTrainerDebugFast
```

This uses the cheaper `2d` config and `custom_trainers/nnUNetTrainerDebugFast.py`,
which cuts training to 5 epochs x 20 iterations. Results from this run are
meaningless — it only confirms data loading, augmentation, loss, and
checkpointing all work before you move to a GPU for the real experiments.
For the actual Phase 1-3 runs, use a GPU (local, or a free tier like Google
Colab / Kaggle notebooks if you don't have one).

## Sanity-check before you trust any of this

Because I couldn't execute this against real data or a real nnunetv2
install, treat this as a **strong first draft**, not verified code:
- Run `python data_prep/prepare_isles26_dataset.py --help` and read the
  assumptions before pointing it at your download.
- After installing custom trainers, run
  `nnUNetv2_train -h` and confirm your new trainer classes show up as valid
  `-tr` options (nnU-Net will error clearly if the class isn't discovered).
- Train one fold of the baseline first before launching all experiments —
  confirms the whole pipeline before you spend GPU-hours on variants.
