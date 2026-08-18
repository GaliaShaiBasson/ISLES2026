# Stroke-lesion segmentation experiments with nnU-Net v2

ISLES'26/ATLAS lesion segmentation: one baseline, a controlled loss study,
lesion-size-aware sampling, and stratified (in-distribution vs.
out-of-distribution site) evaluation, on top of a hardened nnU-Net
operational layer — one cross-platform command instead of the original
platform-specific, multi-script workflow:

```bash
python isles26.py --help
```

The historical project name says ISLES'26; the converter targets the
**ATLAS R3.0 raw** folder layout (native-space, skull-stripped data — see
`CLAUDE.md`). Do not point the converter at a different challenge dataset
without adapting its discovery rules.

See `PROJECT_PLAN.md` for current run status and what's left before the
report is due, and `CLAUDE.md` for the decisions log (*why* things deviate
from nnU-Net/project defaults).

## Setup

Create and activate a Python 3.10+ virtual environment. Install the PyTorch build appropriate for your CPU/GPU first, using PyTorch 2.8.x or lower, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
python isles26.py init --raw-root "/path/to/ATLAS_R3.0_raw"
python isles26.py doctor --create-dirs --require-raw
```

`init` creates a local `.env` and these folders under `workspace/`:

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`
- `evaluation`
- `figures`

The runner loads those paths for each child process. You no longer need permanent shell environment variables, separate Bash/PowerShell commands, or a script that copies trainers into `site-packages`.

## Typical workflow

Generate the train/val/test_id/test_ood split first (replaces nnU-Net's
default unstratified 5-fold CV — see `CLAUDE.md` for why):

```bash
python data_prep/split_dataset.py --raw-root "/path/to/ATLAS_R3.0_raw" --out-dir workspace/splits
```

Do not proceed past this step until `broken_cases.csv` is small/stable.
Then inspect what the converter finds:

```bash
python isles26.py prepare --dry-run
```

Then generate the nnU-Net dataset and metadata:

```bash
python isles26.py prepare
python isles26.py preprocess
```

Run a short pipeline check before spending GPU time:

```bash
python isles26.py train debug
```

The debug command defaults to the `2d` configuration on CPU and produces scientifically meaningless checkpoints. It exists only to catch data-loading and training failures.

Run the real experiments:

```bash
python isles26.py train baseline
python isles26.py train losses
python isles26.py train sampling
```

Useful options:

```bash
python isles26.py train baseline --fold 0 --configuration 3d_fullres
python isles26.py train baseline --continue
python isles26.py train focal-tversky --device cuda
python isles26.py train baseline --print-only
```

`--print-only` prints the underlying nnU-Net command without launching it.

## Evaluation

After nnU-Net writes validation predictions, evaluate one experiment:

```bash
python isles26.py evaluate \
  --pred-dir "/path/to/fold_0/validation" \
  --experiment baseline
```

Repeat for each experiment, then combine and plot:

```bash
python isles26.py aggregate
python isles26.py plot
```

Outputs are written under `workspace/evaluation/` and `workspace/figures/` by default.

## Configuration

Edit `.env` to change dataset ID, fold, nnU-Net storage locations, device, or workspace paths. Relative paths are resolved from the project root. Process-level environment variables override `.env` values.

The custom trainers are discovered through nnU-Net's `nnUNet_extTrainer` mechanism. `isles26.py` sets it automatically to the project root.

## Project layout

```text
isles26.py                         unified runner
.env.example                       reproducible local configuration template
data_prep/prepare_isles26_dataset.py
custom_trainers/                   loss, sampling, and debug trainers
evaluation/                        metrics and aggregation
analysis/                          report figures
training/                          compatibility wrappers for old commands
PROJECT_PLAN.md                    current run status, what's left before the report
CLAUDE.md                          decisions log (why things deviate from defaults)
```

## Important validation limits

The archive did not contain the medical images or trained checkpoints. Static checks and small synthetic tests can validate command construction and pure Python logic, but they cannot establish that the raw-data glob patterns match your exact download or that a full GPU training run converges. Run `prepare --dry-run`, then the debug trainer, before launching the full experiments.
