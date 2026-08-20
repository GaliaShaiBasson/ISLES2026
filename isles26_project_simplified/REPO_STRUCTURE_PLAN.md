# Repo structure cleanup — plan (not yet executed)

Drafted 2026-08-20. Captures the current-state audit and the agreed target
structure so the actual reorganization can happen later without re-deriving
this. Nothing described here has been moved/renamed yet.

## Current state (excluding top-level `archive/`)

### Repo root — `/ISLES2026/`
```
ISLES2026/
├── .claude/
├── data/                          # raw ATLAS downloads (gitignored, 6G)
├── nnUNet_raw/                    # framework storage (gitignored, 3.8G)
├── nnUNet_preprocessed/           # framework storage (gitignored, 13G)
├── nnUNet_results/                # framework storage (gitignored, 29G)
├── environment.yml
├── initial_nb.ipynb                # orphaned — predates the project restructure?
├── README.md                       # one line: "# ISLES2026"
└── isles26_project_simplified/     # <- the actual project, everything below
```

### `isles26_project_simplified/` — source code (fine as-is)
```
analysis/          8 scripts   (plotting, PCA, finalist selection, dice-vs-threshold...)
custom_trainers/   9 files     (nnU-Net trainer subclasses, losses)
data_prep/         4 scripts   (split, prepare, metadata utils)
ensembling/        4 files     (val ensembling/staging)
evaluation/        3 scripts   (metrics, aggregate, postprocess)
tests/             3 files
training/          6 files     (.sh + .ps1 pairs — old, predates isles26.py CLI?)
```

### `isles26_project_simplified/` — loose root files (the problem area)
```
CLAUDE.md, README.md, PROJECT_PLAN.md, PROJECT_REVIEW.md,
ISLES2026_challenge.md, "run instructions"        ← 6 different doc/notes files
General Training Tips....pdf, Project _Guidelines_2026.pdf   ← reference PDFs
isles26.py                                         ← the CLI, fine at root
check_status.sh, sanity_overfit_check.sh,
run_full_experiment.sh, run_500ep_full_experiment.sh,
run_postprocess_grid.sh, queue_postprocess_after_samplingpow.sh  ← 6 shell scripts, no subfolder
.env, requirements.txt
figures/            ← 18 files: CSVs + PNGs, generated output sitting at root
```

### `isles26_project_simplified/workspace/` — runtime output (the real mess)
```
workspace/
├── final_holdout/            2.6G  — locked test set, never touch
├── postprocess_grid/          69M  — ~30 val_search subfolders + test_final, from run_postprocess_grid.sh
├── smoketest_predVal_prob/    17M
├── figures/                  9.8M  — SECOND figures/ dir (this one tracked in git!)
│    ├── finalist_selection/
│    └── augmentation_examples/
├── archive/                  5.9M  — workspace's OWN archive (logs/, broken run backup, stub dirs)
├── postprocess_test/         4.3M
├── evaluation/                2.7M — SECOND evaluation/ dir (results, not code)
│    ├── runs/<run_id>/       — 7 trainer runs, each with results_*.csv + run_manifest.json + archive/
│    ├── archive/             — yet another archive layer, with its OWN runs/ inside
│    └── finalist_selection/
├── sample_run/                512K — isolated smoke-test tree (evaluation/, figures/ inside it too)
├── splits/, splits_full/, splits_sample/   — 3 near-identical split-CSV dirs
├── val_images_staged/, val_images_staged_smoketest/
├── case_metadata_*.csv × 4    — full/dataset001/pow_p05/pow_p05_full, at workspace root
├── *.log × ~9                 — loose training/queue/predict logs at workspace root, 2.2M-876K each
└── *.sh × 4                   — queue_dctopk10.sh, queue_resencm_baseline500.sh, predict_prob_remaining.sh, etc.
```

**Archive nesting found:** `archive/` (repo root) → `workspace/archive/` →
`workspace/evaluation/archive/` → `workspace/evaluation/archive/.../runs/` →
`workspace/evaluation/runs/<run>/archive/`. Five layers of "old stuff lives
here," none documented, none consistent.

## Proposed target structure

```
ISLES2026/
├── nnUNet_raw/ nnUNet_preprocessed/ nnUNet_results/   # unchanged, framework-owned
├── data/                                               # unchanged, raw downloads
├── archive/                                             # ONE archive, repo-wide, gitignored
└── isles26_project_simplified/
    ├── README.md, CLAUDE.md, PROJECT_PLAN.md, PROJECT_REVIEW.md
    ├── docs/
    │   ├── ISLES2026_challenge.md
    │   ├── run_instructions.md            (renamed, extension added)
    │   └── reference/                     (the two PDFs)
    ├── isles26.py, requirements.txt, .env
    ├── scripts/                            (all the root-loose .sh files, one place)
    │   ├── check_status.sh, sanity_overfit_check.sh
    │   ├── run_full_experiment.sh, run_500ep_full_experiment.sh
    │   └── run_postprocess_grid.sh, queue_postprocess_after_samplingpow.sh
    ├── analysis/ custom_trainers/ data_prep/ ensembling/
    ├── evaluation/                          # CODE only (compute_metrics.py etc.)
    ├── training/                            # decide: keep as legacy reference, or fold into scripts/
    ├── tests/
    └── workspace/                           # ALL generated/runtime output, nothing else
        ├── splits/                          # one split tree — retire splits_full vs splits_sample naming confusion, or document the distinction clearly
        ├── case_metadata/                   # the 4 CSVs, grouped
        ├── results/                         # was workspace/evaluation/ — RENAME to avoid echoing evaluation/ code dir
        │   └── runs/<run_id>/
        ├── figures/                         # ONE figures dir, this is it — delete/merge the root-level isles26_project_simplified/figures/
        ├── logs/                            # all the loose *.log files, off the workspace root
        ├── predictions/                     # postprocess_grid/, postprocess_test/, smoketest_predVal_prob/, val_images_staged*/
        ├── sample_run/                      # unchanged — already correctly isolated
        ├── final_holdout/                   # unchanged — locked, do not touch
        └── archive/                         # ONE workspace-level archive, absorbs workspace/evaluation/archive/*
```

### Core moves this implies

1. Delete/merge `isles26_project_simplified/figures/` into `workspace/figures/`
   — kill the duplicate name.
2. Rename `workspace/evaluation/` → `workspace/results/` so it stops
   colliding conceptually with the `evaluation/` *code* dir.
3. Collapse `workspace/archive/` + `workspace/evaluation/archive/` (+ its
   nested `runs/`) into one `workspace/archive/`.
4. Sweep loose root-level `.sh`/`.log` files into `scripts/` and
   `workspace/logs/` respectively.
5. Group the 4 `case_metadata_*.csv` files under `workspace/case_metadata/`.
6. Resolve `initial_nb.ipynb` and `training/`'s `.ps1`/`.sh` pairs — likely
   dead/legacy; confirm before archiving.

## Status

**Directory reorganization done, code references NOT yet updated.**

A full duplicate was made at `../isles26_project/` (untracked, sibling to
this dir) and reorganized into the target layout below. This directory
(`isles26_project_simplified/`) was left untouched throughout — it's still
the working/tracked copy until the switchover.

Applied inside `isles26_project/`:
- `docs/` (+ `docs/reference/` for the two PDFs) — pulled from the loose
  doc files at root (`ISLES2026_challenge.md`, `run_instructions.md`
  renamed from `run instructions`; `PROJECT_PLAN.md`/`PROJECT_REVIEW.md`/
  `README.md`/`CLAUDE.md` stayed at root)
- `scripts/` — core pipeline drivers only: `check_status.sh`,
  `sanity_overfit_check.sh`, `run_full_experiment.sh`,
  `run_500ep_full_experiment.sh`, `run_postprocess_grid.sh`
- Root `figures/` merged into `workspace/figures/`, root copy removed
- `workspace/evaluation/` renamed to `workspace/results/`
- `workspace/archive/` + `workspace/evaluation/archive/` collapsed into one
  `workspace/archive/results_archive/`
- `workspace/case_metadata_*.csv` (4 files) → `workspace/case_metadata/`
- Loose `workspace/*.log` → `workspace/logs/`
- **All** queue/launcher scripts, regardless of prior location, consolidated
  into `workspace/queue_scripts/` (8 files) — these are one-off/disposable
  run-launch helpers, not reusable source, so they live with the other
  runtime/workspace stuff rather than in `scripts/`. Pulled in from
  `scripts/` (`queue_postprocess_after_samplingpow.sh`), `ensembling/`
  (`queue_export_after_resencm.sh`), and the original loose
  `workspace/*.sh` files (`queue_dctopk10.sh`, `queue_resencm_*.sh` x3,
  `queue_baseline1000_after_resencm_export.sh`, `predict_prob_remaining.sh`)
- Prediction/postprocess output dirs (`postprocess_grid/`,
  `postprocess_test/`, `smoketest_predVal_prob/`, `val_images_staged/`,
  `val_images_staged_smoketest/`) → `workspace/predictions/`
- `workspace/figures/` further split by topic instead of flat:
  `learning_curves/`, `results_comparison/`, `threshold_analysis/`
  (alongside the pre-existing `augmentation_examples/` and
  `finalist_selection/`)

- `analysis/finalist_selection/` — grouped `select_finalist_from_val.py`,
  `plot_finalist_selection.py`, `pca_model_redundancy.py` under a subfolder.
  These three are finalist-selection-specific analysis (read existing
  `results_val.csv`s, never touch GPU/CPU); kept them inside `analysis/`
  rather than merging into `ensembling/`, since `ensembling/` actually runs
  predictions and produces new artifacts — a real, worth-preserving
  distinction (`ensemble_val.py`'s own docstring draws this line: "not a
  proxy (the per-case-Dice correlation/PCA in `analysis/`), the actual
  averaged prediction"). The subfolder just makes the relationship between
  the two visible instead of leaving them as unmarked flat files.

Not yet addressed: queue-script `.log` output is still split across
`workspace/logs/` and `workspace/archive/logs/`, not paired up with (or
named to match) the queue script that produced it.

Still open / deliberately left as-is, needs a decision before the
switchover:
- `initial_nb.ipynb` (repo root, outside this project dir) — likely dead,
  confirm before archiving
- `training/` — old `.sh`/`.ps1` pairs, possibly superseded by `isles26.py`
- `workspace/splits/` vs `splits_full/` vs `splits_sample/` naming —
  distinction not obvious from names alone, needs documenting or
  consolidating

## .gitignore

Added a dedicated `--- isles26_project (new, reorganized project root) ---`
section to the repo-root `.gitignore`, mirroring the existing
`isles26_project_simplified/workspace/...` specific-path ignores but updated
for the new layout (`postprocess_test/`, `postprocess_grid/`, and
`val_images_staged` now live under `workspace/predictions/`; `sample_run/`
is unchanged):

```
isles26_project/workspace/predictions/postprocess_test/diceonly250_predTs_cc10
isles26_project/workspace/sample_run/predTs_cc_test
isles26_project/workspace/predictions/postprocess_grid
isles26_project/workspace/predictions/val_images_staged
isles26_project/workspace/predictions/smoketest_predVal_prob
```

`smoketest_predVal_prob/` (two ~8 MB `.npz` probability arrays) was found
missing from this list during a non-ignored-file-size sanity check and added
after, mirrored into the `isles26_project_simplified/` section too. Total
non-ignored size under `isles26_project/` dropped from ~25 MB to ~8.3 MB
after the fix — sane for a git repo.

Everything else in `.gitignore` (`nnUNet_raw/`/`nnUNet_preprocessed/`/
`nnUNet_results/`, `archive/`, `**/final_holdout/`, the bare `figures/`
rule, `.env`, `__pycache__/`, etc.) is unscoped/pattern-based and already
applies to `isles26_project/` without changes — those didn't need mirroring.

Deliberately **kept** the old `isles26_project_simplified/...` ignore lines
rather than replacing them — that directory is still the live, tracked
project until the switchover is actually complete. Remove them once
`isles26_project_simplified/` is retired.

**Not done yet — the actual blocker before `isles26_project/` is usable:**
every hardcoded path in `isles26.py` and the scripts (`workspace/evaluation`,
`workspace/figures` at root, `workspace/case_metadata_*.csv`, etc.) still
points at the *old* locations. Nothing in `isles26_project/` will run
correctly until that reference pass happens. When it does: update
`.gitignore` too (paths like `isles26_project_simplified/workspace/...`
need the `isles26_project/` equivalent), decide whether
`isles26_project_simplified/` gets deleted or kept as a fallback, and
update the README's structure section to match.
