# ISLES'26 Challenge — Data Notes

Source: [isles-26.grand-challenge.org](https://isles-26.grand-challenge.org/), [isles-26.grand-challenge.org/dataset/](https://isles-26.grand-challenge.org/dataset/) (fetched 2026-08-16).

## Challenge overview

- ISLES'26 = generalized stroke infarct segmentation on **T1-weighted MRI**.
- One of the largest annotated stroke infarct datasets to date: ~2,000 annotated scans, sourced from 60+ international centers.
- Covers acute, sub-acute, and chronic disease stages.
- ~40% increase in data diversity and >2x expansion in training data vs. prior ISLES editions.
- Historical project scaffolding in this repo targets the **ATLAS R2.1** raw folder layout; current downloads are **ATLAS R3.0** (`atlas3_training_raw.tar.gz` / `atlas3_training_preprocessed.tar.gz`, AES-256-CBC encrypted, decrypt with OpenSSL — see below).

## Raw vs. preprocessed data — use RAW

**Decision: train and evaluate exclusively on the raw data. Do not use the preprocessed archive for the actual pipeline.**

Direct quotes from the official dataset page:

> "You must download and use the **'Raw Data' folder**. Do not use standardized or normalized versions."

> "The official challenge evaluation will occur exclusively on raw, native-space MRI."

Details:

- **Raw** = original scans in **native clinical space** (original orientation, resolution, coordinate system) — **already skull-stripped**, but not registered/resampled to any standard template.
- **Preprocessed** = the dataset authors' own standardized/normalized version (e.g. registered to a template). No official preprocessed alternative is sanctioned for training in this challenge.
- The **test-set evaluation** (Docker submission on grand-challenge.org) feeds algorithms the **same raw, native-space format** at inference time. Training on the standardized/preprocessed version instead would create a train/test distribution mismatch and likely hurt performance.
- Any additional preprocessing (resampling, intensity normalization, cropping, etc.) is expected to be handled by the participant's own pipeline — this is exactly what nnU-Net's `nnUNetv2_plan_and_preprocess` step does downstream of the raw data.

**Action:** decrypt/extract only `atlas3_training_raw.tar.gz` → `ATLAS_R3.0_raw.tar.gz`. The preprocessed archive can be skipped to save time/disk space unless a non-nnU-Net use case comes up later.

## Decrypting the downloaded archives (Windows)

The ATLAS download provides AES-256-CBC-encrypted `.tar.gz` files; decrypt with OpenSSL.

1. Install OpenSSL for Windows: [slproweb.com/products/Win32OpenSSL.html](https://slproweb.com/products/Win32OpenSSL.html) → **Win64 OpenSSL vX.X.X Light** (EXE installer). SmartScreen may show "Publisher: Unknown" — this is expected for this installer; use "More info" → "Run anyway", or unblock via file Properties.
2. Reopen the terminal so PATH updates take effect. Verify with:
   ```
   openssl version
   ```
   If `openssl` isn't recognized, call it with the full path instead:
   ```
   "C:\Program Files\OpenSSL-Win64\bin\openssl.exe" ...
   ```
3. Decrypt (raw only needed):
   ```
   openssl aes-256-cbc -md sha256 -d -a -in atlas3_training_raw.tar.gz -out ATLAS_R3.0_raw.tar.gz
   ```
   Enter the dataset password when prompted. A "deprecated key derivation used" warning is expected and harmless.
4. Extract:
   ```
   tar -xzf ATLAS_R3.0_raw.tar.gz
   ```

## Pipeline implication for this repo

- Point `isles26.py init --raw-root` (or `data_prep/prepare_isles26_dataset.py --raw-root`) at the extracted **raw** ATLAS folder, matching the `Training_Raw/R0XX/sub-.../ses-.../anat/...` layout the converter expects.
- Let nnU-Net's own `preprocess` step (`nnUNetv2_plan_and_preprocess`, wrapped by `isles26.py preprocess`) generate everything under `nnUNet_preprocessed/` — don't substitute ATLAS's own preprocessed archive for this step.
