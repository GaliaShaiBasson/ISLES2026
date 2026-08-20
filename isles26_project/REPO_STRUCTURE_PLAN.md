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

Not started. When we pick this up: use `git mv` for tracked files, plain
`mv` for untracked/gitignored ones, update `.gitignore` and any hardcoded
paths in `isles26.py`/scripts that reference the old locations (e.g.
`workspace/evaluation`, `workspace/figures`), and update the README's
structure section to match.
