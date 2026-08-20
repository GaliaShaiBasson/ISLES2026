#!/usr/bin/env python3
"""Evaluate/report stage: score predictions, combine result CSVs, and plot figures.

Expected input layout: ``evaluate`` needs a prediction folder (``--pred-dir``,
e.g. nnUNetv2_predict output) and, ideally, ``--trainer`` so results land in
the collision-safe ``workspace/results/runs/<run_id>/`` layout (see
core.compute_run_fingerprint). ``aggregate`` reads whatever ``results_*.csv``
files evaluate produced. ``plot`` reads aggregate's combined ``results.csv``.

What it produces: ``evaluate`` writes ``results_<split>.csv`` plus a
``run_manifest.json`` (fingerprint, git commit, checkpoint path, timestamp)
when run with ``--trainer``. ``aggregate`` writes ``results.csv``,
per-experiment/size-bin/split summaries, and ``runs_index.csv`` (one row per
run, manifest fields flattened). ``plot`` writes report figures via
``analysis/plot_results.py``.

Non-obvious rationale: split out of isles26.py (see core.py's docstring) so
this stage is runnable on its own (``python evaluate_cli.py evaluate
--pred-dir ...``) as well as via the ``isles26.py`` umbrella. ``evaluate``,
``aggregate``, and ``plot`` stay grouped in one stage script (rather than
three separate files) since they're a single linear pipeline over the same
results directory, not independent concerns.

Usage:
    python evaluate_cli.py evaluate --pred-dir ... --trainer nnUNetTrainerBaseline_500epochs --split val
    python evaluate_cli.py aggregate
    python evaluate_cli.py plot
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

from core import (
    PROJECT_ROOT,
    build_environment,
    config_value,
    dataset_folder,
    ensure_workspace,
    run_cli,
    run_command,
)
from run_identity import compute_run_fingerprint, run_id_from_fingerprint, _output_folder


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_run_manifest(run_dir: Path, fingerprint: dict, *, extra: dict) -> None:
    """Provenance record next to a run's result CSVs -- lets later analysis/post-
    processing (and you, months later) answer "what exactly produced this CSV"
    without re-running anything: trainer, epoch budget, sampling weights hash, split
    count, checkpoint identity, git commit, when it was evaluated.
    """
    manifest_path = run_dir / "run_manifest.json"
    manifest = dict(fingerprint)
    manifest.update(extra)
    manifest["git_commit"] = _git_commit()
    manifest["recorded_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_evaluate(args: argparse.Namespace) -> int:
    if not args.trainer and not args.experiment:
        raise SystemExit("evaluate needs either --trainer (recommended) or --experiment (legacy, manual runs).")
    env = build_environment()
    ensure_workspace(env)
    default_gt = Path(env["nnUNet_raw"]) / dataset_folder(env) / "labelsTr"
    gt_dir = Path(args.gt_dir).expanduser().resolve() if args.gt_dir else default_gt
    # Needed regardless of --trainer: also picks the dataset-specific
    # workspace/splits_dataset<id>/manifest.csv fallback below.
    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)

    fingerprint = None
    run_dir = None
    if args.trainer:
        # Collision-safe path: namespace results by the same auto-fingerprint used to
        # guard training output (see compute_run_fingerprint). Two evaluate calls only
        # ever land in the same folder if every hyperparameter that matters (trainer,
        # epoch budget, split fold count, sampling weights content, ...) is identical --
        # so re-running the pipeline with unchanged settings safely lands on the same
        # files (idempotent, no duplication), while changing anything automatically
        # gets its own folder (no silent overwrite of a different run).
        dataset_name = args.dataset_name or config_value(env, "ISLES26_DATASET_NAME")
        configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
        fold = str(args.fold if args.fold is not None else config_value(env, "ISLES26_FOLD"))
        plans_identifier = args.plans or "nnUNetPlans"
        fingerprint = compute_run_fingerprint(
            args.trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier
        )
        rid = run_id_from_fingerprint(fingerprint)
        run_dir = Path(env["ISLES26_RESULTS_DIR"]) / "runs" / rid
        split = args.split or "results"
        # Plans identifier must be part of the default experiment name, not just the
        # trainer name -- nnU-Net namespaces checkpoints by trainer+plans+config+fold
        # (so e.g. Baseline_500epochs on default plans and on ResEnc M plans are
        # genuinely different runs with different results), but experiment_name used
        # to drop plans entirely, so two plans variants of the same trainer collided
        # under one identical "experiment" value in the combined results CSV --
        # aggregate's duplicate-row check (correctly) refused to merge them. Default
        # plans keeps the plain trainer name (no suffix) for backward compatibility
        # with already-written results; any other plans identifier gets appended.
        default_experiment_name = (
            args.trainer if plans_identifier == "nnUNetPlans" else f"{args.trainer}__{plans_identifier}"
        )
        experiment_name = args.experiment or default_experiment_name
        out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else run_dir / f"results_{split}.csv"
        checkpoint_dir = _output_folder(args.trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier)
    else:
        # Legacy/manual path (no --trainer): unchanged flat workspace/results/ layout,
        # for one-off ad-hoc evaluate calls that aren't part of the tracked experiment grid.
        out_dir = Path(env["ISLES26_RESULTS_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = args.experiment.lower().replace(" ", "_").replace("/", "_")
        out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else out_dir / f"results_{slug}.csv"
        experiment_name = args.experiment
        checkpoint_dir = None

    if out_csv.is_file() and not args.print_only:
        if not args.overwrite:
            print(
                f"[skip] {out_csv} already exists"
                + (f" for run {rid}" if run_dir is not None else "")
                + " -- nothing to redo. Pass --overwrite to recompute (archives the old file first)."
            )
            return 0
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = (run_dir if run_dir is not None else out_csv.parent) / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archived = archive_dir / f"{stamp}_{out_csv.name}"
        shutil.move(str(out_csv), str(archived))
        print(f"[archive] Moved previous result aside (not deleted): {archived}")

    if run_dir is not None and not args.print_only:
        run_dir.mkdir(parents=True, exist_ok=True)

    if args.case_metadata_csv:
        case_metadata_csv = args.case_metadata_csv
    else:
        manifest = Path(env["ISLES26_WORKSPACE"]) / f"splits_dataset{dataset_id:03d}" / "manifest.csv"
        if manifest.is_file():
            case_metadata_csv = str(manifest)
        else:
            print(
                f"[warn] {manifest} not found -- falling back to {env['ISLES26_CASE_METADATA_CSV']} "
                "(train+val only; test_id/test_ood cases won't have metadata to join)."
            )
            case_metadata_csv = env["ISLES26_CASE_METADATA_CSV"]

    pred_dir = Path(args.pred_dir).expanduser().resolve()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "compute_metrics.py"),
        "--pred-dir",
        str(pred_dir),
        "--gt-dir",
        str(gt_dir),
        "--case-metadata-csv",
        case_metadata_csv,
        "--experiment-name",
        experiment_name,
        "--out-csv",
        str(out_csv),
        "--lesion-connectivity",
        str(args.lesion_connectivity),
        "--min-lesion-voxels",
        str(args.min_lesion_voxels),
    ]
    run_command(command, env, args.print_only)

    if run_dir is not None and not args.print_only:
        checkpoint_final = checkpoint_dir / "checkpoint_final.pth" if checkpoint_dir else None
        _write_run_manifest(
            run_dir,
            fingerprint,
            extra={
                "run_id": rid,
                "split": split,
                "experiment_name": experiment_name,
                "pred_dir": str(pred_dir),
                "gt_dir": str(gt_dir),
                "case_metadata_csv": case_metadata_csv,
                "checkpoint_final": str(checkpoint_final) if checkpoint_final and checkpoint_final.is_file() else None,
                "checkpoint_final_mtime": (
                    datetime.datetime.fromtimestamp(checkpoint_final.stat().st_mtime).isoformat(timespec="seconds")
                    if checkpoint_final and checkpoint_final.is_file()
                    else None
                ),
            },
        )
    return 0


def _write_runs_index(scan_dir: Path, out_csv: Path) -> None:
    """Catalog every run's manifest into one flat CSV -- trainer, epoch budget,
    sampling weights hash, split fold count, checkpoint identity, git commit, when
    it ran -- so runs can be compared/audited without opening each JSON file or
    re-running anything.
    """
    import csv as csv_module

    manifests = sorted((scan_dir / "runs").glob("*/run_manifest.json"))
    if not manifests:
        return
    rows = []
    fieldnames: list[str] = []
    for path in manifests:
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(row)
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv_module.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} run(s) to {out_csv}")


def cmd_aggregate(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(env["ISLES26_RESULTS_DIR"])
    scan_dir = Path(env["ISLES26_RESULTS_DIR"]) if not args.out_dir else results_dir
    inputs = [Path(p).expanduser().resolve() for p in args.result_csvs]
    if not inputs:
        # Flat legacy layout (workspace/results/results_*.csv) plus the collision-safe
        # nested layout (workspace/results/runs/<run_id>/results_*.csv) -- both are
        # picked up so existing results from before this layout existed keep working.
        inputs = sorted(scan_dir.glob("results_*.csv")) + sorted(scan_dir.glob("runs/*/results_*.csv"))
        inputs = [p for p in inputs if p.name != "results.csv"]
    if not inputs:
        raise SystemExit(f"No per-experiment result CSVs found in {scan_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "aggregate_results.py"),
        *[str(p) for p in inputs],
        "--out-combined",
        str(results_dir / "results.csv"),
        "--out-summary",
        str(results_dir / "summary_by_experiment.csv"),
        "--out-summary-by-size",
        str(results_dir / "summary_by_size_bin.csv"),
        "--out-summary-by-split",
        str(results_dir / "summary_by_split.csv"),
        "--out-summary-by-split-siteweighted",
        str(results_dir / "summary_by_split_siteweighted.csv"),
    ]
    run_command(command, env, args.print_only)
    if not args.print_only:
        _write_runs_index(scan_dir, results_dir / "runs_index.csv")
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_csv = Path(args.results_csv).expanduser().resolve() if args.results_csv else Path(env["ISLES26_RESULTS_DIR"]) / "results.csv"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(env["ISLES26_FIGURES_DIR"])
    command = [
        sys.executable,
        str(PROJECT_ROOT / "analysis" / "plot_results.py"),
        "--results-csv",
        str(results_csv),
        "--out-dir",
        str(out_dir),
    ]
    run_command(command, env, args.print_only)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("evaluate", help="Compute per-case Dice, HD95, and lesion-wise F1 for one prediction folder")
    p.add_argument("--pred-dir", required=True)
    p.add_argument(
        "--trainer",
        default=None,
        help=(
            "Trainer class that produced --pred-dir. Recommended: namespaces output under "
            "workspace/results/runs/<auto-fingerprint>/ (collision-safe across epoch "
            "count, sampling weights, split fold count, etc. -- see CLAUDE.md) and writes "
            "a run_manifest.json with full provenance. Omit only for one-off/manual runs "
            "(falls back to the flat --experiment-named legacy layout)."
        ),
    )
    p.add_argument("--split", default=None, help="val/test/train/etc. -- used with --trainer to name results_<split>.csv")
    p.add_argument("--dataset-id", type=int, help="Used with --trainer; default: ISLES26_DATASET_ID")
    p.add_argument("--dataset-name", help="Used with --trainer; default: ISLES26_DATASET_NAME")
    p.add_argument("--configuration", help="Used with --trainer; default: ISLES26_CONFIGURATION")
    p.add_argument("--fold", help="Used with --trainer; default: ISLES26_FOLD")
    p.add_argument(
        "--plans",
        default=None,
        help="Used with --trainer; must match the --plans the checkpoint was trained with. Default: nnUNetPlans",
    )
    p.add_argument(
        "--experiment",
        default=None,
        help="Label stored in the 'experiment' column / used for the legacy filename. Defaults to --trainer.",
    )
    p.add_argument("--gt-dir")
    p.add_argument("--out-csv")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if the target result CSV already exists (archives the old one first, not deleted)",
    )
    p.add_argument(
        "--case-metadata-csv",
        default=None,
        help=(
            "Defaults to workspace/splits_dataset<id>/manifest.csv (covers all splits: train/val/"
            "test_id/test_ood) if it exists, else falls back to the train+val-only "
            "workspace/case_metadata/case_metadata.csv."
        ),
    )
    p.add_argument("--lesion-connectivity", type=int, default=3, choices=(1, 2, 3),
                    help="scipy connectivity for lesion-wise components (1=6-connected, "
                         "3=26-connected, default 3)")
    p.add_argument("--min-lesion-voxels", type=int, default=0,
                    help="drop connected components smaller than this many voxels before "
                         "lesion-wise matching (default 0 = no filtering)")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("aggregate", help="Combine per-experiment result CSVs")
    p.add_argument("result_csvs", nargs="*")
    p.add_argument("--out-dir", help="Override output dir (default: workspace/results)")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="Generate report figures from aggregated results")
    p.add_argument("--results-csv")
    p.add_argument("--out-dir")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_plot)


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate/report stage: evaluate, aggregate, plot")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


if __name__ == "__main__":
    raise SystemExit(run_cli(build_standalone_parser()))
