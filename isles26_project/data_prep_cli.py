#!/usr/bin/env python3
"""Data prep stage: convert raw ATLAS data to nnU-Net format, then plan/preprocess it.

Expected input layout: raw ATLAS R3.0 data (see ISLES2026_challenge.md) at
``--raw-root`` or ``ISLES26_RAW_ROOT`` in .env; a split manifest directory
(``data_prep/split_dataset.py`` output, default ``workspace/splits_dataset<id>``) for
``prepare`` to restrict imagesTr/labelsTr to train+val only.

What it produces: ``prepare`` writes the nnU-Net raw dataset folder
(imagesTr/labelsTr/imagesTs/labelsTs, dataset.json, splits_final.json) plus a
case-metadata CSV. ``preprocess`` runs nnU-Net planning/preprocessing and
copies the custom splits_final.json into nnUNet_preprocessed so nnU-Net's
own do_split() picks up our stratified split instead of generating one.

Non-obvious rationale: split out of isles26.py (see core.py's docstring) so
this stage is runnable on its own (``python data_prep_cli.py prepare ...``)
as well as via the ``isles26.py`` umbrella. Kept at the project root rather
than moved inside ``data_prep/`` -- that directory holds plain standalone
scripts invoked via subprocess (no ``__init__.py``, not imported as a
package), matching how ``prepare_isles26_dataset.py``/``split_dataset.py``
already work; a root-level CLI module avoids introducing a new import
convention just for this one stage.

Usage:
    python data_prep_cli.py prepare --raw-root /path/to/ATLAS3_Training_Raw
    python data_prep_cli.py preprocess --dataset-id 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core import PROJECT_ROOT, build_environment, config_value, ensure_workspace, run_cli, run_command

#: Even on a very large machine, preprocessing workers each hold a full volume in
#: RAM and parallelism benefit plateaus well before core count does -- this caps
#: the auto-detected process count rather than claiming every core unconditionally.
AUTO_NUM_PROCESSES_CAP = 32
#: Cores left unclaimed for the OS/other work when auto-detecting.
AUTO_NUM_PROCESSES_RESERVE = 2


def detect_num_processes() -> int:
    """Usable CPU count for this machine, right now -- not hardcoded to any specific box.

    Prefers os.sched_getaffinity (reflects cgroup/container CPU limits, e.g. if this
    ever runs sandboxed) over os.cpu_count() (raw hardware count, can overcount in
    that case). Reserves a couple cores and caps the result -- see constants above.
    """
    try:
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms (e.g. macOS)
        n_cpus = os.cpu_count() or 1
    return max(1, min(n_cpus - AUTO_NUM_PROCESSES_RESERVE, AUTO_NUM_PROCESSES_CAP))


def cmd_prepare(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    raw_root = args.raw_root or env.get("ISLES26_RAW_ROOT", "")
    if not raw_root:
        raise SystemExit("Set ISLES26_RAW_ROOT in .env or pass --raw-root.")

    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)

    # --dataset-id 999 is reserved for sample/smoke-test runs (see CLAUDE.md
    # "Sample/smoke-test pipeline runs must never touch real-run artifacts").
    # Without --out-metadata-csv this silently fell back to
    # env["ISLES26_CASE_METADATA_CSV"] -- the SAME shared path a real dataset
    # (e.g. dataset001) uses -- clobbering real sampling-weight metadata with a
    # tiny sample-run subset. Happened for real on 2026-08-18 via
    # sanity_overfit_check.sh's `prepare --dataset-id 999` call; restored from
    # workspace/splits_dataset999/manifest.csv afterward. Hard-refuse instead of silently
    # defaulting, so this fails loudly at the point of the mistake instead of
    # being discovered later as corrupted shared data.
    if dataset_id == 999 and not args.out_metadata_csv:
        raise SystemExit(
            "--dataset-id 999 (the reserved sample/smoke-test dataset) requires an explicit "
            "--out-metadata-csv -- it must never fall back to the shared ISLES26_CASE_METADATA_CSV "
            f"default ({env['ISLES26_CASE_METADATA_CSV']!r}), which is real-run data for another "
            "dataset id. Pass e.g. --out-metadata-csv workspace/sample_run/case_metadata.csv"
        )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "data_prep" / "prepare_isles26_dataset.py"),
        "--raw-root",
        raw_root,
        "--dataset-id",
        str(dataset_id),
        "--dataset-name",
        args.dataset_name or config_value(env, "ISLES26_DATASET_NAME"),
        "--out-metadata-csv",
        args.out_metadata_csv or env["ISLES26_CASE_METADATA_CSV"],
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    if not args.no_split:
        splits_dir = (
            Path(args.splits_dir)
            if args.splits_dir
            else (Path(env["ISLES26_WORKSPACE"]) / f"splits_dataset{dataset_id:03d}")
        )
        if splits_dir.is_dir():
            command += ["--splits-dir", str(splits_dir)]
        elif args.splits_dir:
            raise SystemExit(f"--splits-dir does not exist: {splits_dir}")
        else:
            print(
                f"[warn] {splits_dir} not found -- writing ALL discovered cases to imagesTr "
                "(no train/val/test split applied). Run data_prep/split_dataset.py first, "
                "or pass --no-split to silence this warning."
            )
    run_command(command, env, args.print_only)
    return 0


def _copy_splits_final_json(env: dict[str, str], dataset_id: int, dataset_name: str) -> None:
    """Carry our custom splits_final.json (written by prepare) into nnUNet_preprocessed.

    nnU-Net's do_split() only looks for splits_final.json inside
    nnUNet_preprocessed/<dataset>/, which nnUNetv2_plan_and_preprocess creates --
    it doesn't exist yet when `prepare` runs, so the file has to be copied over
    here, after preprocessing, rather than written directly by `prepare`.
    """
    import shutil

    raw_dataset_dir = Path(env["nnUNet_raw"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    source = raw_dataset_dir / "splits_final.json"
    if not source.is_file():
        return  # prepare was run with --no-split / without a splits-dir; nothing to carry over
    preprocessed_dataset_dir = Path(env["nnUNet_preprocessed"]) / f"Dataset{dataset_id:03d}_{dataset_name}"
    if not preprocessed_dataset_dir.is_dir():
        print(f"[warn] {preprocessed_dataset_dir} not found; could not install splits_final.json")
        return
    target = preprocessed_dataset_dir / "splits_final.json"
    shutil.copy2(source, target)
    print(f"Installed custom splits_final.json: {target}")


def cmd_preprocess(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)
    dataset_name = config_value(env, "ISLES26_DATASET_NAME")

    plans_marker = Path(env["nnUNet_preprocessed"]) / f"Dataset{dataset_id:03d}_{dataset_name}" / "nnUNetPlans.json"
    if plans_marker.is_file() and not args.overwrite and not args.print_only:
        raise SystemExit(
            f"Preprocessed data already exists: {plans_marker.parent}. Pass --overwrite to redo "
            "planning/preprocessing (e.g. after a data or split change), or skip this step."
        )

    env_override = env.get("ISLES26_PREPROCESS_NUM_PROCESSES")
    num_processes = args.num_processes or (int(env_override) if env_override else None) or detect_num_processes()
    print(f"Using {num_processes} processes for preprocessing/fingerprint extraction (detected/configured for this machine)")

    command = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "-np", str(num_processes),
        "-npfp", str(num_processes),
    ]
    if not args.no_verify:
        command.append("--verify_dataset_integrity")
    if args.overwrite:
        command.append("--clean")
    run_command(command, env, args.print_only)
    if not args.print_only:
        _copy_splits_final_json(env, dataset_id, dataset_name)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("prepare", help="Convert raw ATLAS data to nnU-Net format")
    p.add_argument("--raw-root")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing generated dataset folder")
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "--splits-dir",
        default=None,
        help="Split manifest dir from split_dataset.py (default: workspace/splits_dataset<id> if it exists)",
    )
    p.add_argument(
        "--out-metadata-csv",
        default=None,
        help="Override output path (default: workspace/case_metadata/case_metadata.csv)",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help="Ignore workspace/splits_dataset<id> and write every discovered case to imagesTr (legacy behavior)",
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("preprocess", help="Run nnU-Net planning and preprocessing")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "--overwrite", action="store_true", help="Redo planning/preprocessing even if it already exists"
    )
    p.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help=(
            "Processes for -np/-npfp. Default: auto-detected from this machine's usable "
            "CPU count (see ISLES26_PREPROCESS_NUM_PROCESSES in .env to set a fixed value instead)."
        ),
    )
    p.set_defaults(func=cmd_preprocess)


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data prep stage: prepare and preprocess")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


if __name__ == "__main__":
    raise SystemExit(run_cli(build_standalone_parser()))
