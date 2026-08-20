#!/usr/bin/env python3
"""Resolve a trainer(+plans) identifier to its val-set probability directory.

Expected input layout: none read directly -- this module only computes paths
under the shared `nnUNet_results/Dataset002_ATLAS/` tree and checks for real
`.npz` files there. Trainer identifiers may be a bare trainer class name or
`"trainer (PlansName)"` (matching `select_finalist_from_val.discover_runs`'s
own key format exactly), so a trainer copied from that script's printed
output resolves to the right nnU-Net folder without hand-editing.

What it produces: nothing written -- pure path resolution, shared by
`ensembling/ensemble_val.py` (real ensemble scoring) and
`analysis/pca_probability_redundancy.py`/`analysis/voxel_probability_
correlation.py` (read-only probability analysis), so both agree on where a
trainer's val probabilities actually live without duplicating the
resolution logic.

Non-obvious rationale: prefers the manually-exported `predVal_prob/`
(written by `ensembling/export_val_probabilities.sh`/its queue-script
variants), but falls back to nnU-Net's own automatic `fold_0/validation/`
folder if that one has the real per-case `.npz` probabilities instead --
concretely, standard-plans `nnUNetTrainerBaseline_500epochs`, which was
never in `export_val_probabilities.sh`'s trainer list but already has real
val-set probabilities from an earlier `--val --npz` validate-only pass
(verified 2026-08-20: 120/120 cases, matching `splits_full/manifest.csv`
exactly, newer than `checkpoint_final.pth` -- not stale). Checking for real
`.npz` files (not just directory existence) also catches an
existing-but-empty `predVal_prob/` correctly rather than trusting it.
"""
from __future__ import annotations

from pathlib import Path

NNUNET_RESULTS = Path("/home/galia/ISLES2026/nnUNet_results")
DATASET_NAME = "Dataset002_ATLAS"
DEFAULT_PLANS = "nnUNetPlans"


def parse_trainer_key(key: str) -> tuple[str, str]:
    """Split a `"trainer"` or `"trainer (PlansName)"` key into
    (trainer_class_name, plans_identifier)."""
    if key.endswith(")") and " (" in key:
        trainer, plans = key.rsplit(" (", 1)
        return trainer, plans[:-1]
    return key, DEFAULT_PLANS


def predval_dir(key: str) -> Path:
    """Resolve a trainer(+plans) key to its val-probability directory --
    `predVal_prob/` if it has real `.npz` files, else `validation/` if that
    one does instead, else `predVal_prob/` again (so a caller checking
    `.exists()`/`glob` on the result correctly sees nothing there)."""
    trainer, plans = parse_trainer_key(key)
    base = NNUNET_RESULTS / DATASET_NAME / f"{trainer}__{plans}__3d_fullres" / "fold_0"
    exported = base / "predVal_prob"
    if any(exported.glob("*.npz")):
        return exported
    fallback = base / "validation"
    if any(fallback.glob("*.npz")):
        return fallback
    return exported  # neither has real data; check_predval_dirs reports this as missing


def check_predval_dirs(trainers: list[str]) -> dict[str, Path]:
    """Resolve every trainer in `trainers`, printing which source (manual
    export vs. nnU-Net's automatic validation pass) each one used. Raises if
    any trainer has no real `.npz` probabilities anywhere, listing all
    missing trainers at once rather than failing on the first."""
    dirs = {}
    missing = []
    for t in trainers:
        d = predval_dir(t)
        if not any(d.glob("*.npz")):
            missing.append(t)
            continue
        dirs[t] = d
        source = "predVal_prob/ (manual export)" if d.name == "predVal_prob" \
            else "validation/ (nnU-Net's automatic val pass, no predVal_prob/ export exists)"
        print(f"  {t}: using {source} -> {d}")
    if missing:
        raise SystemExit(
            "Missing val-set probabilities (no .npz in either predVal_prob/ or validation/) for: "
            + ", ".join(missing) +
            "\nRun ensembling/export_val_probabilities.sh for these trainers first."
        )
    return dirs
