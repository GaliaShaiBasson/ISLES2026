#!/usr/bin/env python3
"""Cross-platform command runner for the ISLES/ATLAS nnU-Net project.

Run ``python isles26.py --help`` for the available commands.
The runner loads project settings from ``.env`` and passes them only to the
child nnU-Net processes, so users do not need to configure permanent shell
variables or run platform-specific scripts.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"
EXPECTED_NNUNET_VERSION = "2.8.1"

PATH_KEYS = {
    "ISLES26_RAW_ROOT",
    "ISLES26_WORKSPACE",
    "ISLES26_CASE_METADATA_CSV",
    "ISLES26_RESULTS_DIR",
    "ISLES26_FIGURES_DIR",
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
}

DEFAULTS = {
    "ISLES26_RAW_ROOT": "",
    "ISLES26_WORKSPACE": "workspace",
    "ISLES26_CASE_METADATA_CSV": "workspace/case_metadata.csv",
    "ISLES26_RESULTS_DIR": "workspace/evaluation",
    "ISLES26_FIGURES_DIR": "workspace/figures",
    "ISLES26_DATASET_ID": "1",
    "ISLES26_DATASET_NAME": "ATLAS",
    "ISLES26_CONFIGURATION": "3d_fullres",
    "ISLES26_FOLD": "0",
    "ISLES26_DEVICE": "cuda",
    "ISLES26_NUM_GPUS": "1",
    "nnUNet_raw": "workspace/nnUNet_raw",
    "nnUNet_preprocessed": "workspace/nnUNet_preprocessed",
    "nnUNet_results": "workspace/nnUNet_results",
}

TRAINER_GROUPS = {
    # All real-experiment trainers below share a 250-epoch budget (down from
    # nnU-Net's 1000-epoch default) so the study fits the available GPU time.
    # Every condition uses the same budget so the baseline/loss/sampling
    # comparison stays controlled. See PROJECT_PLAN.md.
    "baseline": ["nnUNetTrainerBaseline_250epochs"],
    "losses": [
        "nnUNetTrainerDiceOnly_250epochs",
        "nnUNetTrainerFocal_250epochs",
        "nnUNetTrainerTversky_250epochs",
        "nnUNetTrainerFocalTversky_250epochs",
    ],
    "dice": ["nnUNetTrainerDiceOnly_250epochs"],
    "focal": ["nnUNetTrainerFocal_250epochs"],
    "tversky": ["nnUNetTrainerTversky_250epochs"],
    "focal-tversky": ["nnUNetTrainerFocalTversky_250epochs"],
    "sampling": ["nnUNetTrainerLesionAwareSampling_250epochs"],
    "debug": ["nnUNetTrainerDebugFast"],
}


# TRAINER_GROUPS = {
#     "baseline": ["nnUNetTrainer"],
#     "losses": [
#         "nnUNetTrainerDiceOnly",
#         "nnUNetTrainerFocal",
#         "nnUNetTrainerTversky",
#         "nnUNetTrainerFocalTversky",
#     ],
#     "dice": ["nnUNetTrainerDiceOnly"],
#     "focal": ["nnUNetTrainerFocal"],
#     "tversky": ["nnUNetTrainerTversky"],
#     "focal-tversky": ["nnUNetTrainerFocalTversky"],
#     "sampling": ["nnUNetTrainerLesionAwareSampling"],
#     "debug": ["nnUNetTrainerDebugFast"],
# }


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: missing key")
        values[key] = _strip_quotes(value)
    return values


def _resolve_path(value: str) -> str:
    if not value:
        return ""
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return str(expanded.resolve())


def build_environment() -> dict[str, str]:
    config = dict(DEFAULTS)
    config.update(load_dotenv())

    # Explicit process environment variables take precedence over .env.
    for key in set(config) | PATH_KEYS | {"nnUNet_extTrainer"}:
        if key in os.environ:
            config[key] = os.environ[key]

    for key in PATH_KEYS:
        if key in config:
            config[key] = _resolve_path(config[key])

    env = os.environ.copy()
    env.update(config)

    # nnU-Net 2.8.1 can discover trainers outside site-packages. Point it at
    # the project root so custom_trainers is importable as a package.
    existing = env.get("nnUNet_extTrainer", "")
    trainer_root = str(PROJECT_ROOT)
    entries = [p for p in existing.split(os.pathsep) if p]
    if trainer_root not in entries:
        entries.insert(0, trainer_root)
    env["nnUNet_extTrainer"] = os.pathsep.join(entries)
    return env


def config_value(env: dict[str, str], key: str, cast=str):
    try:
        return cast(env[key])
    except KeyError as exc:
        raise RuntimeError(f"Missing configuration value: {key}") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid value for {key}: {env.get(key)!r}") from exc


def print_command(command: Iterable[str]) -> None:
    print("$", shlex.join([str(part) for part in command]))


def run_command(command: list[str], env: dict[str, str], dry_run: bool = False) -> None:
    print_command(command)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def dataset_folder(env: dict[str, str]) -> str:
    dataset_id = config_value(env, "ISLES26_DATASET_ID", int)
    dataset_name = config_value(env, "ISLES26_DATASET_NAME")
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def ensure_workspace(env: dict[str, str]) -> None:
    for key in (
        "ISLES26_WORKSPACE",
        "ISLES26_RESULTS_DIR",
        "ISLES26_FIGURES_DIR",
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)


def write_env_file(raw_root: str | None, force: bool) -> None:
    if ENV_FILE.exists() and not force:
        print(f"Keeping existing {ENV_FILE}. Use --force to replace it.")
        return
    values = dict(DEFAULTS)
    if raw_root:
        values["ISLES26_RAW_ROOT"] = str(Path(raw_root).expanduser().resolve())
    lines = [
        "# Local project configuration. Paths may be absolute or project-relative.",
        "# This file is loaded by isles26.py; no permanent shell variables are required.",
        "",
    ]
    for key, value in values.items():
        lines.append(f'{key}="{value}"')
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {ENV_FILE}")


def cmd_init(args: argparse.Namespace) -> int:
    write_env_file(args.raw_root, args.force)
    env = build_environment()
    ensure_workspace(env)
    print(f"Workspace: {env['ISLES26_WORKSPACE']}")
    print("Next: edit .env if needed, then run `python isles26.py doctor`.")
    return 0


def _status(ok: bool, label: str, detail: str = "") -> None:
    marker = "OK" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{marker:4}] {label}{suffix}")


def cmd_doctor(args: argparse.Namespace) -> int:
    env = build_environment()
    failures = 0

    python_ok = sys.version_info >= (3, 10)
    _status(python_ok, "Python", sys.version.split()[0])
    failures += int(not python_ok)

    for key in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        path = Path(env[key])
        if args.create_dirs:
            path.mkdir(parents=True, exist_ok=True)
        ok = path.is_dir()
        _status(ok, key, str(path))
        failures += int(not ok)

    for command in ("nnUNetv2_plan_and_preprocess", "nnUNetv2_train"):
        location = shutil.which(command, path=env.get("PATH"))
        ok = location is not None
        _status(ok, command, location or "not on PATH")
        failures += int(not ok)

    try:
        installed = importlib.metadata.version("nnunetv2")
        version_ok = installed == EXPECTED_NNUNET_VERSION
        _status(version_ok, "nnunetv2 version", f"installed {installed}; expected {EXPECTED_NNUNET_VERSION}")
        failures += int(not version_ok)
    except importlib.metadata.PackageNotFoundError:
        _status(False, "nnunetv2 package", "not installed")
        failures += 1
        installed = None

    try:
        torch_version = importlib.metadata.version("torch")
        torch_ok = tuple(int(p) for p in torch_version.split("+")[0].split(".")[:2]) <= (2, 8)
        _status(torch_ok, "PyTorch version", f"installed {torch_version}; nnU-Net recommends 2.8.x or lower")
        failures += int(not torch_ok)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        _status(False, "PyTorch package", "not installed or version unreadable")
        failures += 1

    raw_root = env.get("ISLES26_RAW_ROOT", "")
    raw_ok = bool(raw_root) and Path(raw_root).is_dir()
    _status(raw_ok, "Raw dataset root", raw_root or "set ISLES26_RAW_ROOT in .env")
    if args.require_raw:
        failures += int(not raw_ok)

    if installed:
        discovery_code = (
            "from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f;"
            "names=['nnUNetTrainerDebugFast','nnUNetTrainerFocalTversky_250epochs',"
            "'nnUNetTrainerLesionAwareSampling_250epochs'];"
            "[f(n) for n in names];print('custom trainer discovery OK')"
        )
        result = subprocess.run(
            [sys.executable, "-c", discovery_code],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        ok = result.returncode == 0
        detail = result.stdout.strip().splitlines()[-1] if ok and result.stdout.strip() else result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        _status(ok, "Custom trainer discovery", detail)
        failures += int(not ok)

    if failures:
        print(f"\nDoctor found {failures} blocking issue(s).")
        return 1
    print("\nEnvironment is ready.")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    raw_root = args.raw_root or env.get("ISLES26_RAW_ROOT", "")
    if not raw_root:
        raise SystemExit("Set ISLES26_RAW_ROOT in .env or pass --raw-root.")

    command = [
        sys.executable,
        str(PROJECT_ROOT / "data_prep" / "prepare_isles26_dataset.py"),
        "--raw-root",
        raw_root,
        "--dataset-id",
        str(args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)),
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
        splits_dir = Path(args.splits_dir) if args.splits_dir else (Path(env["ISLES26_WORKSPACE"]) / "splits")
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


def _copy_splits_final_json(env: dict[str, str], dataset_id: int, dataset_name: str) -> None:
    """Carry our custom splits_final.json (written by prepare) into nnUNet_preprocessed.

    nnU-Net's do_split() only looks for splits_final.json inside
    nnUNet_preprocessed/<dataset>/, which nnUNetv2_plan_and_preprocess creates --
    it doesn't exist yet when `prepare` runs, so the file has to be copied over
    here, after preprocessing, rather than written directly by `prepare`.
    """
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


def _find_latest_checkpoint(
    trainer: str, dataset_id: int, dataset_name: str, configuration: str, fold: str, env: dict[str, str]
) -> Path | None:
    checkpoint = _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env) / "checkpoint_latest.pth"
    return checkpoint if checkpoint.is_file() else None


def _train_one(
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    device: str,
    num_gpus: int,
    args: argparse.Namespace,
    env: dict[str, str],
) -> None:
    continue_training = args.continue_training
    if not continue_training and not args.overwrite and not args.validate_only:
        latest = _find_latest_checkpoint(trainer, dataset_id, dataset_name, configuration, fold, env)
        if latest is not None:
            # Training was interrupted mid-run (crash, kill, machine restart) and left a
            # partial checkpoint_latest.pth. nnU-Net's own default (no --c) would silently
            # start over from epoch 0 and eventually overwrite it -- for unattended
            # multi-hour runs that's a real risk, not a hypothetical, so resume by default
            # instead. --overwrite explicitly opts back into a genuine restart.
            print(f"[resume] Found partial checkpoint, continuing from it instead of restarting: {latest}")
            continue_training = True

    command = ["nnUNetv2_train", str(dataset_id), configuration, fold]
    if trainer != "nnUNetTrainer":
        command += ["-tr", trainer]
    if num_gpus != 1:
        command += ["-num_gpus", str(num_gpus)]
    if device != "cuda":
        command += ["-device", device]
    if continue_training:
        command.append("--c")
    if args.validate_only:
        command.append("--val")
    if args.val_best:
        command.append("--val_best")
    if args.npz:
        command.append("--npz")
    if args.disable_checkpointing:
        command.append("--disable_checkpointing")
    run_command(command, env, args.print_only)


def _output_folder(
    trainer: str, dataset_id: int, dataset_name: str, configuration: str, fold: str, env: dict[str, str]
) -> Path:
    return (
        Path(env["nnUNet_results"])
        / f"Dataset{dataset_id:03d}_{dataset_name}"
        / f"{trainer}__nnUNetPlans__{configuration}"
        / f"fold_{fold}"
    )


def _find_existing_checkpoint(
    trainer: str, dataset_id: int, dataset_name: str, configuration: str, fold: str, env: dict[str, str]
) -> Path | None:
    checkpoint = _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env) / "checkpoint_final.pth"
    return checkpoint if checkpoint.is_file() else None


def cmd_train(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)
    dataset_name = args.dataset_name or config_value(env, "ISLES26_DATASET_NAME")
    fold = str(args.fold if args.fold is not None else config_value(env, "ISLES26_FOLD"))

    if args.experiment == "debug":
        configuration = args.configuration or "2d"
        device = args.device or "cpu"
    else:
        configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
        device = args.device or config_value(env, "ISLES26_DEVICE")
    num_gpus = args.num_gpus or config_value(env, "ISLES26_NUM_GPUS", int)

    if args.preprocess:
        preprocess_args = argparse.Namespace(
            dataset_id=dataset_id,
            no_verify=False,
            print_only=args.print_only,
            overwrite=False,
        )
        cmd_preprocess(preprocess_args)

    trainers = [args.trainer] if args.trainer else TRAINER_GROUPS[args.experiment]
    if any(trainer.startswith("nnUNetTrainerLesionAwareSampling") for trainer in trainers):
        metadata = Path(env["ISLES26_CASE_METADATA_CSV"])
        if not metadata.is_file() and not args.print_only:
            raise SystemExit(f"Sampling metadata is missing: {metadata}. Run `python isles26.py prepare` first.")

    is_group_run = args.trainer is None and len(trainers) > 1
    skip_guard_active = (
        not args.continue_training and not args.validate_only and not args.overwrite and not args.print_only
    )

    for trainer in trainers:
        print(f"\n== {trainer} ==")
        if skip_guard_active:
            existing = _find_existing_checkpoint(trainer, dataset_id, dataset_name, configuration, fold, env)
            if existing is not None:
                message = (
                    f"Training already completed: {existing}. Pass --overwrite to restart from "
                    "scratch, or --continue to resume/verify."
                )
                if is_group_run:
                    print(f"[skip] {message}")
                    continue
                raise SystemExit(message)
        _train_one(trainer, dataset_id, dataset_name, configuration, fold, device, num_gpus, args, env)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    default_gt = Path(env["nnUNet_raw"]) / dataset_folder(env) / "labelsTr"
    gt_dir = Path(args.gt_dir).expanduser().resolve() if args.gt_dir else default_gt
    out_dir = Path(env["ISLES26_RESULTS_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.experiment.lower().replace(" ", "_").replace("/", "_")
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else out_dir / f"results_{slug}.csv"

    if args.case_metadata_csv:
        case_metadata_csv = args.case_metadata_csv
    else:
        manifest = Path(env["ISLES26_WORKSPACE"]) / "splits" / "manifest.csv"
        if manifest.is_file():
            case_metadata_csv = str(manifest)
        else:
            print(
                f"[warn] {manifest} not found -- falling back to {env['ISLES26_CASE_METADATA_CSV']} "
                "(train+val only; test_id/test_ood cases won't have metadata to join)."
            )
            case_metadata_csv = env["ISLES26_CASE_METADATA_CSV"]

    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "compute_metrics.py"),
        "--pred-dir",
        str(Path(args.pred_dir).expanduser().resolve()),
        "--gt-dir",
        str(gt_dir),
        "--case-metadata-csv",
        case_metadata_csv,
        "--experiment-name",
        args.experiment,
        "--out-csv",
        str(out_csv),
        "--lesion-connectivity",
        str(args.lesion_connectivity),
        "--min-lesion-voxels",
        str(args.min_lesion_voxels),
    ]
    run_command(command, env, args.print_only)
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(env["ISLES26_RESULTS_DIR"])
    scan_dir = Path(env["ISLES26_RESULTS_DIR"]) if not args.out_dir else results_dir
    inputs = [Path(p).expanduser().resolve() for p in args.result_csvs]
    if not inputs:
        inputs = sorted(scan_dir.glob("results_*.csv"))
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
    ]
    run_command(command, env, args.print_only)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One cross-platform entry point for data preparation, nnU-Net training, and evaluation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create .env and local workspace folders")
    p.add_argument("--raw-root", help="Raw ATLAS R2.1 dataset root")
    p.add_argument("--force", action="store_true", help="Replace an existing .env")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", help="Validate dependencies, paths, and custom trainer discovery")
    p.add_argument("--create-dirs", action="store_true", help="Create missing nnU-Net storage folders")
    p.add_argument("--require-raw", action="store_true", help="Treat a missing raw dataset root as blocking")
    p.set_defaults(func=cmd_doctor)

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
        help="Split manifest dir from split_dataset.py (default: workspace/splits if it exists)",
    )
    p.add_argument(
        "--out-metadata-csv",
        default=None,
        help="Override output path (default: workspace/case_metadata.csv)",
    )
    p.add_argument(
        "--no-split",
        action="store_true",
        help="Ignore workspace/splits and write every discovered case to imagesTr (legacy behavior)",
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

    p = sub.add_parser("train", help="Train one experiment or a predefined experiment group")
    p.add_argument("experiment", choices=sorted(TRAINER_GROUPS))
    p.add_argument("--trainer", help="Override the predefined trainer class")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name", help="Override the dataset name (default: ISLES26_DATASET_NAME in .env)")
    p.add_argument("--fold")
    p.add_argument("--configuration")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"])
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--preprocess", action="store_true", help="Run preprocessing before training")
    p.add_argument("--continue", dest="continue_training", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--val-best", action="store_true")
    p.add_argument("--npz", action="store_true")
    p.add_argument("--disable-checkpointing", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Restart training from scratch even if a completed checkpoint_final.pth already exists",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Compute per-case Dice, HD95, and lesion-wise F1 for one prediction folder")
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--experiment", required=True)
    p.add_argument("--gt-dir")
    p.add_argument("--out-csv")
    p.add_argument(
        "--case-metadata-csv",
        default=None,
        help=(
            "Defaults to workspace/splits/manifest.csv (covers all splits: train/val/"
            "test_id/test_ood) if it exists, else falls back to the train+val-only "
            "workspace/case_metadata.csv."
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
    p.add_argument("--out-dir", help="Override output dir (default: workspace/evaluation)")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="Generate report figures from aggregated results")
    p.add_argument("--results-csv")
    p.add_argument("--out-dir")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_plot)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args) or 0)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
