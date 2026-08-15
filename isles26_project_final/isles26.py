#!/usr/bin/env python3
"""Cross-platform runner for the ATLAS/nnU-Net lesion-size experiments.

Use ``python isles26.py --help`` for commands. Project-local ``.env`` values
are preferred over inherited process variables so an old Windows/Linux shell
setting cannot silently override this project's configuration.
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
    "ISLES26_METADATA_DIR",
    "ISLES26_RESULTS_DIR",
    "ISLES26_FIGURES_DIR",
    "ISLES26_PREDICTIONS_DIR",
    "nnUNet_raw",
    "nnUNet_preprocessed",
    "nnUNet_results",
}

DEFAULTS = {
    "ISLES26_RAW_ROOT": "",
    "ISLES26_WORKSPACE": "workspace",
    "ISLES26_METADATA_DIR": "workspace/metadata",
    "ISLES26_RESULTS_DIR": "workspace/evaluation",
    "ISLES26_FIGURES_DIR": "workspace/figures",
    "ISLES26_PREDICTIONS_DIR": "workspace/predictions",
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

EXPERIMENT_TRAINERS = {
    "baseline": "nnUNetTrainer",
    "focal": "nnUNetTrainerFocal",
    "tversky": "nnUNetTrainerTversky",
    "sampling": "nnUNetTrainerLesionAwareSampling",
    "curriculum": "nnUNetTrainerCurriculumLesionAwareSampling",
    "debug": "nnUNetTrainerDebugFast",
}
REAL_EXPERIMENTS = ["baseline", "focal", "tversky", "sampling", "curriculum"]


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_dotenv(path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
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
    # Inherited values are accepted when .env has no corresponding setting.
    for key in set(DEFAULTS) | {"nnUNet_extTrainer"}:
        if key in os.environ:
            config[key] = os.environ[key]
    # Project-local configuration intentionally wins over inherited values.
    config.update(load_dotenv())

    for key in PATH_KEYS:
        if key in config:
            config[key] = _resolve_path(config[key])

    env = os.environ.copy()
    env.update(config)

    # nnU-Net 2.8.1 supports trainer discovery outside site-packages.
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


def run_command(command: list[str], env: dict[str, str], print_only: bool = False) -> None:
    print_command(command)
    if not print_only:
        subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _dataset_id_name(env: dict[str, str], dataset_id: int | None, dataset_name: str | None) -> tuple[int, str]:
    return (
        dataset_id if dataset_id is not None else config_value(env, "ISLES26_DATASET_ID", int),
        dataset_name or config_value(env, "ISLES26_DATASET_NAME"),
    )


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def metadata_csv_path(env: dict[str, str], dataset_id: int, dataset_name: str) -> Path:
    return Path(env["ISLES26_METADATA_DIR"]) / f"{dataset_folder_name(dataset_id, dataset_name)}_case_metadata.csv"


def prediction_dir(env: dict[str, str], dataset_id: int, dataset_name: str, experiment: str) -> Path:
    return Path(env["ISLES26_PREDICTIONS_DIR"]) / dataset_folder_name(dataset_id, dataset_name) / experiment


def ensure_workspace(env: dict[str, str]) -> None:
    for key in (
        "ISLES26_WORKSPACE",
        "ISLES26_METADATA_DIR",
        "ISLES26_RESULTS_DIR",
        "ISLES26_FIGURES_DIR",
        "ISLES26_PREDICTIONS_DIR",
        "nnUNet_raw",
        "nnUNet_preprocessed",
        "nnUNet_results",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)


def _write_env(values: dict[str, str]) -> None:
    lines = [
        "# Project-local configuration for isles26.py.",
        "# Values in this file take precedence over inherited shell variables.",
        "",
    ]
    for key in DEFAULTS:
        lines.append(f"{key}={values.get(key, DEFAULTS[key])}")
    ENV_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_init(args: argparse.Namespace) -> int:
    if args.force or not ENV_FILE.exists():
        values = dict(DEFAULTS)
        action = "Created" if not ENV_FILE.exists() else "Reset"
    else:
        values = dict(DEFAULTS)
        values.update(load_dotenv())
        action = "Updated"

    updates: dict[str, str] = {}
    if args.raw_root is not None:
        updates["ISLES26_RAW_ROOT"] = str(Path(args.raw_root).expanduser().resolve())
    if args.dataset_id is not None:
        updates["ISLES26_DATASET_ID"] = str(args.dataset_id)
    if args.dataset_name is not None:
        updates["ISLES26_DATASET_NAME"] = args.dataset_name
    if args.configuration is not None:
        updates["ISLES26_CONFIGURATION"] = args.configuration
    if args.fold is not None:
        updates["ISLES26_FOLD"] = str(args.fold)
    if args.device is not None:
        updates["ISLES26_DEVICE"] = args.device
    values.update(updates)

    _write_env(values)
    env = build_environment()
    ensure_workspace(env)
    print(f"{action} {ENV_FILE}")
    print(f"Raw root: {env['ISLES26_RAW_ROOT'] or '(not set)'}")
    print(f"Default dataset: {dataset_folder_name(config_value(env, 'ISLES26_DATASET_ID', int), env['ISLES26_DATASET_NAME'])}")
    print(f"Workspace: {env['ISLES26_WORKSPACE']}")
    return 0


def _status(ok: bool, label: str, detail: str = "") -> None:
    marker = "OK" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{marker:4}] {label}{suffix}")


def _torch_version_supported(version: str) -> bool:
    base = version.split("+")[0]
    parts = base.split(".")
    major_minor = tuple(int(p) for p in parts[:2])
    return major_minor >= (2, 1) and major_minor != (2, 9)


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

    for command in ("nnUNetv2_plan_and_preprocess", "nnUNetv2_train", "nnUNetv2_predict"):
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
        torch_ok = _torch_version_supported(torch_version)
        detail = f"installed {torch_version}; nnU-Net 2.8.1 requires torch>=2.1.2 and excludes 2.9.*"
        _status(torch_ok, "PyTorch version", detail)
        failures += int(not torch_ok)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        _status(False, "PyTorch package", "not installed or version unreadable")
        failures += 1

    raw_root = env.get("ISLES26_RAW_ROOT", "")
    raw_ok = bool(raw_root) and Path(raw_root).is_dir()
    _status(raw_ok, "Raw dataset root", raw_root or "set it with `python isles26.py init --raw-root ...`")
    if args.require_raw:
        failures += int(not raw_ok)

    dataset_id, dataset_name = _dataset_id_name(env, None, None)
    metadata = metadata_csv_path(env, dataset_id, dataset_name)
    if metadata.is_file():
        _status(True, "Metadata for default dataset", str(metadata))
    else:
        print(f"[NOTE] Metadata for default dataset — {metadata} (created by prepare)")

    if installed:
        discovery_code = (
            "from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f;"
            "names=['nnUNetTrainerDebugFast','nnUNetTrainerFocal','nnUNetTrainerTversky',"
            "'nnUNetTrainerLesionAwareSampling','nnUNetTrainerCurriculumLesionAwareSampling'];"
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
        if ok and result.stdout.strip():
            detail = result.stdout.strip().splitlines()[-1]
        elif result.stderr.strip():
            detail = result.stderr.strip().splitlines()[-1]
        else:
            detail = ""
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
    raw_root = str(Path(args.raw_root).expanduser().resolve()) if args.raw_root else env.get("ISLES26_RAW_ROOT", "")
    if not raw_root:
        raise SystemExit("Set the raw root with `python isles26.py init --raw-root ...` or pass --raw-root.")

    dataset_id, dataset_name = _dataset_id_name(env, args.dataset_id, args.dataset_name)
    metadata = metadata_csv_path(env, dataset_id, dataset_name)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "data_prep" / "prepare_isles26_dataset.py"),
        "--raw-root", raw_root,
        "--dataset-id", str(dataset_id),
        "--dataset-name", dataset_name,
        "--out-metadata-csv", str(metadata),
    ]
    if args.dry_run:
        command.append("--dry-run")
    if args.overwrite:
        command.append("--overwrite")
    run_command(command, env, args.print_only)
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id, _ = _dataset_id_name(env, args.dataset_id, None)
    command = ["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id)]
    if not args.no_verify:
        command.append("--verify_dataset_integrity")
    run_command(command, env, args.print_only)
    return 0


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
    command = ["nnUNetv2_train", str(dataset_id), configuration, fold]
    if trainer != "nnUNetTrainer":
        command += ["-tr", trainer]
    if num_gpus != 1:
        command += ["-num_gpus", str(num_gpus)]
    if device != "cuda":
        command += ["-device", device]
    if args.continue_training:
        command.append("--c")
    if args.validate_only:
        command.append("--val")
    if args.val_best:
        command.append("--val_best")
    if args.npz:
        command.append("--npz")
    if args.disable_checkpointing:
        command.append("--disable_checkpointing")

    child_env = env.copy()
    child_env["ISLES26_CASE_METADATA_CSV"] = str(metadata_csv_path(env, dataset_id, dataset_name))
    run_command(command, child_env, args.print_only)


def cmd_train(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id, dataset_name = _dataset_id_name(env, args.dataset_id, args.dataset_name)
    fold = str(args.fold if args.fold is not None else config_value(env, "ISLES26_FOLD"))

    if args.experiment == "debug":
        configuration = args.configuration or "2d"
        device = args.device or "cpu"
    else:
        configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
        device = args.device or config_value(env, "ISLES26_DEVICE")
    num_gpus = args.num_gpus or config_value(env, "ISLES26_NUM_GPUS", int)

    if args.preprocess:
        cmd_preprocess(argparse.Namespace(dataset_id=dataset_id, no_verify=False, print_only=args.print_only))

    experiments = REAL_EXPERIMENTS if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        trainer = args.trainer if args.trainer and len(experiments) == 1 else EXPERIMENT_TRAINERS[experiment]
        if trainer in {"nnUNetTrainerLesionAwareSampling", "nnUNetTrainerCurriculumLesionAwareSampling"}:
            metadata = metadata_csv_path(env, dataset_id, dataset_name)
            if not metadata.is_file() and not args.print_only:
                raise SystemExit(f"Sampling metadata is missing: {metadata}. Run prepare for this dataset first.")
        print(f"\n== {experiment}: {trainer} ==")
        _train_one(trainer, dataset_id, dataset_name, configuration, fold, device, num_gpus, args, env)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id, dataset_name = _dataset_id_name(env, args.dataset_id, args.dataset_name)
    configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
    device = args.device or config_value(env, "ISLES26_DEVICE")
    trainer = EXPERIMENT_TRAINERS[args.experiment]
    folds = args.folds or [str(config_value(env, "ISLES26_FOLD"))]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else prediction_dir(
        env, dataset_id, dataset_name, args.experiment
    )
    command = [
        "nnUNetv2_predict",
        "-i", str(Path(args.input_dir).expanduser().resolve()),
        "-o", str(out_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-f", *[str(fold) for fold in folds],
        "-chk", args.checkpoint,
        "-device", device,
    ]
    if trainer != "nnUNetTrainer":
        command += ["-tr", trainer]
    if args.save_probabilities:
        command.append("--save_probabilities")
    if args.continue_prediction:
        command.append("--continue_prediction")
    if args.disable_tta:
        command.append("--disable_tta")
    run_command(command, env, args.print_only)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    dataset_id, dataset_name = _dataset_id_name(env, args.dataset_id, args.dataset_name)
    default_gt = Path(env["nnUNet_raw"]) / dataset_folder_name(dataset_id, dataset_name) / "labelsTr"
    gt_dir = Path(args.gt_dir).expanduser().resolve() if args.gt_dir else default_gt
    pred_dir = Path(args.pred_dir).expanduser().resolve() if args.pred_dir else prediction_dir(
        env, dataset_id, dataset_name, args.experiment
    )
    out_dir = Path(env["ISLES26_RESULTS_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = args.experiment.lower().replace(" ", "_").replace("/", "_")
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else out_dir / f"results_{slug}.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "compute_metrics.py"),
        "--pred-dir", str(pred_dir),
        "--gt-dir", str(gt_dir),
        "--case-metadata-csv", str(metadata_csv_path(env, dataset_id, dataset_name)),
        "--experiment-name", args.experiment,
        "--out-csv", str(out_csv),
    ]
    run_command(command, env, args.print_only)
    return 0


def _aggregate_command(env: dict[str, str], inputs: list[Path]) -> list[str]:
    results_dir = Path(env["ISLES26_RESULTS_DIR"])
    return [
        sys.executable,
        str(PROJECT_ROOT / "evaluation" / "aggregate_results.py"),
        *[str(p) for p in inputs],
        "--out-combined", str(results_dir / "results.csv"),
        "--out-summary", str(results_dir / "summary_by_experiment.csv"),
        "--out-summary-by-size", str(results_dir / "summary_by_size_bin.csv"),
        "--out-summary-by-center", str(results_dir / "summary_by_center.csv"),
        "--out-summary-by-size-center", str(results_dir / "summary_by_size_and_center.csv"),
    ]


def cmd_aggregate(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_dir = Path(env["ISLES26_RESULTS_DIR"])
    inputs = [Path(p).expanduser().resolve() for p in args.result_csvs]
    if not inputs:
        inputs = [p for p in sorted(results_dir.glob("results_*.csv")) if p.name != "results.csv"]
    if not inputs:
        raise SystemExit(f"No per-experiment result CSVs found in {results_dir}")
    run_command(_aggregate_command(env, inputs), env, args.print_only)
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_csv = Path(args.results_csv).expanduser().resolve() if args.results_csv else Path(env["ISLES26_RESULTS_DIR"]) / "results.csv"
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else Path(env["ISLES26_FIGURES_DIR"])
    command = [
        sys.executable,
        str(PROJECT_ROOT / "analysis" / "plot_results.py"),
        "--results-csv", str(results_csv),
        "--out-dir", str(out_dir),
    ]
    run_command(command, env, args.print_only)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    env = build_environment()
    ensure_workspace(env)
    results_dir = Path(env["ISLES26_RESULTS_DIR"])
    inputs = [p for p in sorted(results_dir.glob("results_*.csv")) if p.name != "results.csv"]
    if not inputs:
        raise SystemExit(f"No per-experiment result CSVs found in {results_dir}")
    run_command(_aggregate_command(env, inputs), env, args.print_only)
    plot_args = argparse.Namespace(results_csv=None, out_dir=None, print_only=args.print_only)
    return cmd_plot(plot_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One cross-platform entry point for ATLAS data preparation, nnU-Net experiments, prediction, and analysis."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="Create or update project-local configuration")
    p.add_argument("--raw-root", help="Raw ATLAS R2.1 dataset root")
    p.add_argument("--dataset-id", type=int, help="Default nnU-Net dataset ID")
    p.add_argument("--dataset-name", help="Default nnU-Net dataset name")
    p.add_argument("--configuration", help="Default nnU-Net configuration")
    p.add_argument("--fold", help="Default fold")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], help="Default device")
    p.add_argument("--force", action="store_true", help="Reset all settings to defaults before applying supplied options")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", help="Validate dependencies, paths, and trainer discovery")
    p.add_argument("--create-dirs", action="store_true")
    p.add_argument("--require-raw", action="store_true")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("prepare", help="Convert ATLAS data to nnU-Net format and create metadata")
    p.add_argument("--raw-root")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("preprocess", help="Run nnU-Net planning and preprocessing")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("train", help="Train baseline, focal, Tversky, fixed sampling, curriculum, or all")
    p.add_argument("experiment", choices=REAL_EXPERIMENTS + ["all", "debug"])
    p.add_argument("--trainer", help="Advanced: override trainer class for a single experiment")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name")
    p.add_argument("--fold")
    p.add_argument("--configuration")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"])
    p.add_argument("--num-gpus", type=int)
    p.add_argument("--preprocess", action="store_true")
    p.add_argument("--continue", dest="continue_training", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--val-best", action="store_true")
    p.add_argument("--npz", action="store_true")
    p.add_argument("--disable-checkpointing", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Run nnU-Net inference for one trained experiment")
    p.add_argument("experiment", choices=REAL_EXPERIMENTS)
    p.add_argument("--input-dir", required=True, help="nnU-Net-formatted input images (_0000.nii.gz)")
    p.add_argument("--out-dir")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name")
    p.add_argument("--configuration")
    p.add_argument("--folds", nargs="+", help="One or more folds; default is configured fold")
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"])
    p.add_argument("--save-probabilities", action="store_true")
    p.add_argument("--continue-prediction", action="store_true")
    p.add_argument("--disable-tta", action="store_true")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="Compute per-case Dice/HD95 and attach lesion-size/center metadata")
    p.add_argument("--experiment", required=True)
    p.add_argument("--pred-dir", help="Defaults to workspace/predictions/<dataset>/<experiment>")
    p.add_argument("--dataset-id", type=int)
    p.add_argument("--dataset-name")
    p.add_argument("--gt-dir")
    p.add_argument("--out-csv")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("aggregate", help="Combine result CSVs and create overall/size/center summaries")
    p.add_argument("result_csvs", nargs="*")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("plot", help="Generate overall, size, center, and size×center figures")
    p.add_argument("--results-csv")
    p.add_argument("--out-dir")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("analyze", help="Convenience command: aggregate all result CSVs and generate figures")
    p.add_argument("--print-only", action="store_true")
    p.set_defaults(func=cmd_analyze)
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
