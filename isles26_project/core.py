"""Shared infrastructure for every isles26 stage CLI (setup_cli.py, data_prep_cli.py,
train_cli.py, evaluate_cli.py) and the isles26.py umbrella that wires them together.

Expected input layout: none directly -- this module has no CLI of its own. It's
imported by every stage script, each a sibling file in the project root.

What it produces: nothing on its own. Provides .env/environment loading
(``build_environment``), subprocess execution helpers (``run_command``), the
run-identity fingerprinting system that keeps ``train``/``evaluate`` from
silently colliding across hyperparameter changes (``compute_run_fingerprint``
and friends), and shared checkpoint/output-folder path resolution.

Non-obvious rationale: this module exists because ``isles26.py`` used to be a
single ~1,300-line file mixing this shared infrastructure with five distinct
command groups (setup, data prep, train, evaluate/report). Splitting those
groups into independently-runnable stage scripts (so each can also be driven
from its own .sh) would have meant duplicating this infrastructure --
especially the fingerprinting logic, which is exactly the kind of duplicated-
script drift the project's "Foundational hardening pass" (see CLAUDE.md)
already fixed once. Pulling it out here instead means every stage script
imports the same environment/fingerprint code rather than re-implementing it.
``run_cli`` is the one further piece of shared boilerplate (argparse dispatch
+ exception handling) so every stage script's ``if __name__ == "__main__":``
block is one line.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
    "ISLES26_CASE_METADATA_CSV": "workspace/case_metadata/case_metadata.csv",
    "ISLES26_RESULTS_DIR": "workspace/results",
    "ISLES26_FIGURES_DIR": "workspace/figures/results_comparison",
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


SAMPLING_METADATA_BY_TRAINER = {
    # trainer class name -> (env var it reads, default path) for the sampling-weight CSV
    # it loads at train time. Keyed by trainer so the fingerprint below can hash the
    # actual weights a given run used, not just guess from the trainer name.
    "nnUNetTrainerLesionAwareSampling_250epochs": (
        "ISLES26_CASE_METADATA_CSV",
        "workspace/case_metadata/case_metadata.csv",
    ),
    "nnUNetTrainerLesionAwareSamplingPow_250epochs": (
        "ISLES26_SAMPLING_POW_METADATA_CSV",
        "workspace/case_metadata/case_metadata_pow_p05.csv",
    ),
    "nnUNetTrainerLesionAwareSampling_500epochs_full": (
        "ISLES26_CASE_METADATA_CSV_FULL",
        "workspace/case_metadata/case_metadata_full.csv",
    ),
    "nnUNetTrainerLesionAwareSamplingPow_500epochs_full": (
        "ISLES26_SAMPLING_POW_METADATA_CSV_FULL",
        "workspace/case_metadata/case_metadata_pow_p05_full.csv",
    ),
    # Curriculum trainer computes weights dynamically from size_bin/lesion_volume_mm3
    # at a p that changes over training (see its docstring) -- it does NOT read this
    # CSV's sampling_weight column. This hash therefore only tracks identity of the
    # underlying per-case volume/bin data, not the full p-schedule (which is fixed in
    # the class definition itself, so a schedule change requires a new class/name).
    "nnUNetTrainerLesionAwareSamplingPowCurriculum_500epochs_full": (
        "ISLES26_SAMPLING_POW_CURRICULUM_METADATA_CSV_FULL",
        "workspace/case_metadata/case_metadata_full.csv",
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
    regenerating workspace/case_metadata/case_metadata_pow_p05.csv with a different --p (same
    filename) is correctly seen as a different run, without requiring a new
    trainer class or a manually-remembered CLI tag.
    """
    if trainer not in SAMPLING_METADATA_BY_TRAINER:
        return None

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
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    env: dict[str, str],
    plans_identifier: str = "nnUNetPlans",
) -> dict:
    """Everything that actually determines this run's results, beyond what nnU-Net's own
    output-folder naming (trainer/plans/config/fold) captures on its own.

    This is the single source of truth for run identity used both to guard against
    silently reusing/overwriting an nnU-Net checkpoint folder for an incompatible config
    (see _guard_run_fingerprint) and to namespace our own results/evaluation output
    (see run_id_from_fingerprint) -- so "does this collide" is answered the same way in
    both places instead of two hand-maintained schemes drifting apart.

    plans_identifier is deliberately omitted from the returned dict when it's the
    historical default ("nnUNetPlans") -- every fingerprint computed before --plans
    existed (including markers already written on disk by in-flight runs) has no
    "plans" key, so keeping the default silent here means this stays byte-identical
    to the pre-existing hash/run_id for every run that never asked for a non-default
    plans identifier. Only a non-default plans (e.g. nnUNetResEncUNetMPlans) adds the
    key -- which is exactly what's needed to keep it from colliding with a
    same-trainer default-plans run.
    """
    introspected = _introspect_trainer(trainer, env)
    fingerprint = {
        "trainer": trainer,
        "dataset": f"Dataset{dataset_id:03d}_{dataset_name}",
        "configuration": configuration,
        "fold": fold,
        "n_folds_in_split": _splits_final_n_folds(dataset_id, dataset_name, env),
        "num_epochs": introspected["num_epochs"],
        "save_every": introspected["save_every"],
        "sampling_weight_hash": _sampling_weight_hash(trainer, env),
    }
    if plans_identifier != "nnUNetPlans":
        fingerprint["plans"] = plans_identifier
    return fingerprint


def run_id_from_fingerprint(fingerprint: dict) -> str:
    """Short, human-browsable, collision-safe id for a run: readable prefix + content hash.

    The hash (not the prefix) is what actually guarantees safety -- two runs only ever
    share a run_id if every field in compute_run_fingerprint() is identical. The prefix
    only gains a plans segment when the fingerprint has a non-default "plans" key (see
    compute_run_fingerprint) -- default-plans run_ids are unchanged from before --plans
    existed.
    """
    digest = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    fold = fingerprint["fold"]
    plans_part = f"__{fingerprint['plans']}" if "plans" in fingerprint else ""
    return f"{fingerprint['trainer']}{plans_part}__{fingerprint['configuration']}__fold{fold}__fp{digest}"


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


def _output_folder(
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    env: dict[str, str],
    plans_identifier: str = "nnUNetPlans",
) -> Path:
    return (
        Path(env["nnUNet_results"])
        / f"Dataset{dataset_id:03d}_{dataset_name}"
        / f"{trainer}__{plans_identifier}__{configuration}"
        / f"fold_{fold}"
    )


def _find_latest_checkpoint(
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    env: dict[str, str],
    plans_identifier: str = "nnUNetPlans",
) -> Path | None:
    checkpoint = (
        _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier)
        / "checkpoint_latest.pth"
    )
    return checkpoint if checkpoint.is_file() else None


def _find_existing_checkpoint(
    trainer: str,
    dataset_id: int,
    dataset_name: str,
    configuration: str,
    fold: str,
    env: dict[str, str],
    plans_identifier: str = "nnUNetPlans",
) -> Path | None:
    checkpoint = (
        _output_folder(trainer, dataset_id, dataset_name, configuration, fold, env, plans_identifier)
        / "checkpoint_final.pth"
    )
    return checkpoint if checkpoint.is_file() else None


def run_cli(parser: argparse.ArgumentParser) -> int:
    """Shared argparse-dispatch + exception-handling wrapper for every stage script's
    ``if __name__ == "__main__":`` block (and the isles26.py umbrella's ``main()``),
    so each one is a single line instead of duplicating this try/except.
    """
    args = parser.parse_args()
    try:
        return int(args.func(args) or 0)
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.", file=sys.stderr)
        return exc.returncode or 1
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
