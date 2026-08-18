#!/usr/bin/env python3
"""Cross-platform command runner for the ISLES/ATLAS nnU-Net project.

Run ``python isles26.py --help`` for the available commands.
The runner loads project settings from ``.env`` and passes them only to the
child nnU-Net processes, so users do not need to configure permanent shell
variables or run platform-specific scripts.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
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
    "tversky-mild": ["nnUNetTrainerTverskyMild_250epochs"],
    "sampling": ["nnUNetTrainerLesionAwareSampling_250epochs"],
    "sampling-pow": ["nnUNetTrainerLesionAwareSamplingPow_250epochs"],
    "debug": ["nnUNetTrainerDebugFast"],

    # 500-epoch run against Dataset002_ATLAS (the corrected, 1,453-case full
    # split -- see CLAUDE.md). Requires --dataset-id 2. Sampling uses the
    # original 4:2:1 bin-level ratio, not the power-law variant -- the pow
    # variant's own 250-epoch results (val/test_id/test_ood Dice all lower
    # than plain 4:2:1) argued against it before this run was staged. Includes
    # tversky-mild alongside focal-tversky (in addition to, not instead of --
    # tversky-mild's own 250-epoch/dataset001 run had no results yet when this
    # was assembled). DA5 augmentation was investigated but not verified in
    # time; stock augmentation only for this run.
    "baseline-500": ["nnUNetTrainerBaseline_500epochs"],
    "focal-tversky-500": ["nnUNetTrainerFocalTversky_500epochs"],
    "tversky-mild-500": ["nnUNetTrainerTverskyMild_500epochs"],
    "sampling-500": ["nnUNetTrainerLesionAwareSampling_500epochs_full"],

    # Widened-intensity-augmentation variant, built on the plain baseline for a
    # clean A/B against baseline-500 (see custom_trainers/nnUNetTrainerWideAug.py
    # and CLAUDE.md/PROJECT_PLAN.md/PROJECT_REVIEW.md "two runs remain" entries).
    "baseline-wideaug-500": ["nnUNetTrainerWideAugBaseline_500epochs"],

    # Power-law (p=0.5) bin-level sampling at 500ep/dataset002 -- re-tested on the
    # test_ood angle specifically, not the val-Dice angle that argued against it at
    # 250ep/dataset001 (see nnUNetTrainerLesionAwareSamplingPow_500epochs_full's
    # docstring and PROJECT_PLAN.md). Queued to launch after baseline-wideaug-500.
    "sampling-pow-500": ["nnUNetTrainerLesionAwareSamplingPow_500epochs_full"],

    # Overfit-a-tiny-subset sanity gate (training-tips checklist item 1 / this
    # project's own bug history -- see PROJECT_REVIEW.md). Run against a new
    # trainer on Dataset999_ATLASsample (small sample dataset) BEFORE launching
    # it for real -- see sanity_overfit_check.sh. Not a real experiment result.
    "overfit-check-wideaug": ["nnUNetTrainerWideAugBaseline_OverfitCheck"],
    # Stock-augmentation control for the same check -- see
    # custom_trainers/nnUNetTrainerOverfitCheck.py:nnUNetTrainerBaseline_OverfitCheck.
    "overfit-check-baseline": ["nnUNetTrainerBaseline_OverfitCheck"],
    # Gate for sampling-pow-500 -- needs ISLES26_SAMPLING_POW_METADATA_CSV_FULL
    # overridden to a Dataset999-specific pow CSV, see
    # nnUNetTrainerOverfitCheck.py:nnUNetTrainerLesionAwareSamplingPow_OverfitCheck.
    "overfit-check-samplingpow": ["nnUNetTrainerLesionAwareSamplingPow_OverfitCheck"],
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

    dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)

    # --dataset-id 999 is reserved for sample/smoke-test runs (see CLAUDE.md
    # "Sample/smoke-test pipeline runs must never touch real-run artifacts").
    # Without --out-metadata-csv this silently fell back to
    # env["ISLES26_CASE_METADATA_CSV"] -- the SAME shared path a real dataset
    # (e.g. dataset001) uses -- clobbering real sampling-weight metadata with a
    # tiny sample-run subset. Happened for real on 2026-08-18 via
    # sanity_overfit_check.sh's `prepare --dataset-id 999` call; restored from
    # workspace/splits/manifest.csv afterward. Hard-refuse instead of silently
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


SAMPLING_METADATA_BY_TRAINER = {
    # trainer class name -> (env var it reads, default path) for the sampling-weight CSV
    # it loads at train time. Keyed by trainer so the fingerprint below can hash the
    # actual weights a given run used, not just guess from the trainer name.
    "nnUNetTrainerLesionAwareSampling_250epochs": (
        "ISLES26_CASE_METADATA_CSV",
        "workspace/case_metadata.csv",
    ),
    "nnUNetTrainerLesionAwareSamplingPow_250epochs": (
        "ISLES26_SAMPLING_POW_METADATA_CSV",
        "workspace/case_metadata_pow_p05.csv",
    ),
    "nnUNetTrainerLesionAwareSampling_500epochs_full": (
        "ISLES26_CASE_METADATA_CSV_FULL",
        "workspace/case_metadata_full.csv",
    ),
    "nnUNetTrainerLesionAwareSamplingPow_500epochs_full": (
        "ISLES26_SAMPLING_POW_METADATA_CSV_FULL",
        "workspace/case_metadata_pow_p05_full.csv",
    ),
}


def _splits_final_n_folds(dataset_id: int, dataset_name: str, env: dict[str, str]) -> int | str:
    """Number of folds in the splits_final.json actually driving this dataset's training.

    Not derivable from trainer/config/fold alone -- e.g. switching from our usual
    single-fold split to a real 5-fold CV split reuses the same trainer/config/fold=0
    for fold 0 of each, which would otherwise land in the identical nnU-Net output
    folder. Read from nnUNet_preprocessed (what training actually uses), not the raw
    copy, since that's the file do_split() reads.
    """
    path = Path(env["nnUNet_preprocessed"]) / f"Dataset{dataset_id:03d}_{dataset_name}" / "splits_final.json"
    if not path.is_file():
        return "unknown"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return len(data) if isinstance(data, list) else "unknown"
    except (json.JSONDecodeError, OSError):
        return "unknown"


def _sampling_weight_hash(trainer: str, env: dict[str, str]) -> str | None:
    """Short hash of a sampling trainer's actual (case_id, sampling_weight) pairs.

    Ties run identity to the real weights content, not just a file path -- so
    regenerating workspace/case_metadata_pow_p05.csv with a different --p (same
    filename) is correctly seen as a different run, without requiring a new
    trainer class or a manually-remembered CLI tag.
    """
    if trainer not in SAMPLING_METADATA_BY_TRAINER:
        return None
    import hashlib

    env_var, default_path = SAMPLING_METADATA_BY_TRAINER[trainer]
    csv_path = Path(env.get(env_var, default_path))
    if not csv_path.is_file():
        return "missing"
    import csv as csv_module

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv_module.DictReader(handle)
        rows = sorted((row["case_id"], row["sampling_weight"]) for row in reader)
    digest = hashlib.sha1(repr(rows).encode("utf-8")).hexdigest()
    return digest[:10]


_TRAINER_INTROSPECT_CODE = """
import inspect, json, re, sys
from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name
cls = recursive_find_trainer_class_by_name(sys.argv[1])
info = {"num_epochs": None, "save_every": None}
try:
    src = inspect.getsource(cls.__init__)
except (OSError, TypeError):
    src = ""
m = re.search(r"self\\.num_epochs\\s*=\\s*(\\d+)", src)
if m:
    info["num_epochs"] = int(m.group(1))
m = re.search(r"self\\.save_every\\s*=\\s*(\\d+)", src)
if m:
    info["save_every"] = int(m.group(1))
print(json.dumps(info))
"""


def _introspect_trainer(trainer: str, env: dict[str, str]) -> dict:
    """Best-effort source-level introspection of a trainer's num_epochs/save_every.

    Regex over inspect.getsource(cls.__init__) rather than instantiating the class --
    real instantiation needs plans/dataset_json/fold that aren't available at this
    point in the CLI. Catches the exact "hypothetical 500-epoch trainer" gotcha from
    CLAUDE.md's decisions log without relying on the trainer's class *name* to say so:
    if num_epochs can't be found in source (e.g. inherited unchanged), falls back to
    nnU-Net's documented default of 1000.
    """
    result = subprocess.run(
        [sys.executable, "-c", _TRAINER_INTROSPECT_CODE, trainer],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"num_epochs": "unknown", "save_every": "unknown"}
    try:
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"num_epochs": "unknown", "save_every": "unknown"}
    if info.get("num_epochs") is None:
        info["num_epochs"] = 1000  # nnU-Net's own default when a trainer never overrides it
    if info.get("save_every") is None:
        info["save_every"] = 50  # nnU-Net's own default
    return info


def compute_run_fingerprint(
    trainer: str, dataset_id: int, dataset_name: str, configuration: str, fold: str, env: dict[str, str]
) -> dict:
    """Everything that actually determines this run's results, beyond what nnU-Net's own
    output-folder naming (trainer/plans/config/fold) captures on its own.

    This is the single source of truth for run identity used both to guard against
    silently reusing/overwriting an nnU-Net checkpoint folder for an incompatible config
    (see _guard_run_fingerprint) and to namespace our own results/evaluation output
    (see run_id_from_fingerprint) -- so "does this collide" is answered the same way in
    both places instead of two hand-maintained schemes drifting apart.
    """
    introspected = _introspect_trainer(trainer, env)
    return {
        "trainer": trainer,
        "dataset": f"Dataset{dataset_id:03d}_{dataset_name}",
        "configuration": configuration,
        "fold": fold,
        "n_folds_in_split": _splits_final_n_folds(dataset_id, dataset_name, env),
        "num_epochs": introspected["num_epochs"],
        "save_every": introspected["save_every"],
        "sampling_weight_hash": _sampling_weight_hash(trainer, env),
    }


def run_id_from_fingerprint(fingerprint: dict) -> str:
    """Short, human-browsable, collision-safe id for a run: readable prefix + content hash.

    The hash (not the prefix) is what actually guarantees safety -- two runs only ever
    share a run_id if every field in compute_run_fingerprint() is identical.
    """
    import hashlib

    digest = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    fold = fingerprint["fold"]
    return f"{fingerprint['trainer']}__{fingerprint['configuration']}__fold{fold}__fp{digest}"


def _fingerprint_diff(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [f"{k}: {old.get(k)!r} -> {new.get(k)!r}" for k in keys if old.get(k) != new.get(k)]


def _guard_run_fingerprint(output_folder: Path, fingerprint: dict, overwrite: bool) -> None:
    """Refuse to let a changed hyperparameter silently reuse/overwrite an nnU-Net
    checkpoint folder that nnU-Net itself would consider "the same" (identical
    trainer/plans/config/fold), unless --overwrite is passed.

    Without this, e.g. regenerating case_metadata_pow_p05.csv with a different --p, or
    switching from a 1-fold to a 5-fold split, would resume/overwrite checkpoints for a
    run with genuinely different settings -- nnU-Net's own folder naming can't see the
    difference, only compute_run_fingerprint() can.
    """
    marker = output_folder / "isles26_fingerprint.json"
    if not marker.is_file():
        # First time training this exact nnU-Net folder (or a pre-fingerprint legacy
        # run, e.g. tonight's overnight run): nothing to compare against -- adopt the
        # current fingerprint as the folder's identity going forward.
        output_folder.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    try:
        stored = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        stored = {}

    diff = _fingerprint_diff(stored, fingerprint)
    if not diff:
        return  # identical run -- normal resume/rerun, proceed as before

    if not overwrite:
        raise SystemExit(
            f"{output_folder} already holds a run with different settings than what you're "
            f"about to train (nnU-Net would silently reuse/overwrite it, since trainer/"
            f"config/fold match). Changed field(s):\n  " + "\n  ".join(diff) +
            "\nPass --overwrite to archive the old run and start fresh, or use a trainer "
            "class name that doesn't collide with an existing one."
        )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = output_folder.parent / f"{output_folder.name}.archived_{stamp}"
    shutil.move(str(output_folder), str(archived))
    print(f"[archive] Moved conflicting run aside (not deleted): {archived}")
    output_folder.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    # Each lesion-aware-sampling variant reads its own metadata CSV (distinct env var
    # + default path per trainer -- see custom_trainers/nnUNetTrainerLesionAwareSampling.py)
    # so concurrent variants never share or collide over one file. Check whichever file
    # the specific trainer(s) being launched will actually read, not always the original.
    for trainer_name in trainers:
        if trainer_name not in SAMPLING_METADATA_BY_TRAINER:
            continue
        env_var, default_path = SAMPLING_METADATA_BY_TRAINER[trainer_name]
        metadata = Path(env.get(env_var, default_path))
        if not metadata.is_file() and not args.print_only:
            raise SystemExit(
                f"Sampling metadata is missing for {trainer_name}: {metadata} "
                f"(set via {env_var}). Generate it first."
            )

    is_group_run = args.trainer is None and len(trainers) > 1
    skip_guard_active = (
        not args.continue_training and not args.validate_only and not args.overwrite and not args.print_only
    )

    for trainer in trainers:
        print(f"\n== {trainer} ==")
        output_folder = _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env)
        if not args.print_only:
            # Runs before the checkpoint skip-guard below: a hyperparameter change that
            # nnU-Net's own folder naming can't see (num_epochs, sampling weights
            # content, split fold count, ...) must be caught even when no
            # checkpoint_final.pth exists yet (e.g. a stale partial run from a since-
            # changed config). See compute_run_fingerprint/_guard_run_fingerprint.
            fingerprint = compute_run_fingerprint(trainer, dataset_id, dataset_name, configuration, fold, env)
            _guard_run_fingerprint(output_folder, fingerprint, args.overwrite)
            print(f"[run_id] {run_id_from_fingerprint(fingerprint)}")
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
        dataset_id = args.dataset_id or config_value(env, "ISLES26_DATASET_ID", int)
        dataset_name = args.dataset_name or config_value(env, "ISLES26_DATASET_NAME")
        configuration = args.configuration or config_value(env, "ISLES26_CONFIGURATION")
        fold = str(args.fold if args.fold is not None else config_value(env, "ISLES26_FOLD"))
        fingerprint = compute_run_fingerprint(args.trainer, dataset_id, dataset_name, configuration, fold, env)
        rid = run_id_from_fingerprint(fingerprint)
        run_dir = Path(env["ISLES26_RESULTS_DIR"]) / "runs" / rid
        split = args.split or "results"
        experiment_name = args.experiment or args.trainer
        out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else run_dir / f"results_{split}.csv"
        checkpoint_dir = _output_folder(args.trainer, dataset_id, dataset_name, configuration, fold, env)
    else:
        # Legacy/manual path (no --trainer): unchanged flat workspace/evaluation/ layout,
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

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

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
        # Flat legacy layout (workspace/evaluation/results_*.csv) plus the collision-safe
        # nested layout (workspace/evaluation/runs/<run_id>/results_*.csv) -- both are
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
    p.add_argument(
        "--trainer",
        default=None,
        help=(
            "Trainer class that produced --pred-dir. Recommended: namespaces output under "
            "workspace/evaluation/runs/<auto-fingerprint>/ (collision-safe across epoch "
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
