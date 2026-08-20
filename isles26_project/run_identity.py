"""Run-identity fingerprinting: keeps train/evaluate from silently colliding across
hyperparameter changes that nnU-Net's own output-folder naming can't see.

Expected input layout: none directly -- this module has no CLI of its own. It's
imported only by ``train_cli.py`` and ``evaluate_cli.py`` (never
``setup_cli.py``/``data_prep_cli.py``, which have no notion of run identity).

What it produces: nothing on its own. Provides ``compute_run_fingerprint``
(everything that determines a run's results, beyond trainer/plans/config/fold),
``run_id_from_fingerprint`` (a short collision-safe id used to namespace result
output), ``_guard_run_fingerprint`` (refuses to silently reuse/overwrite an
incompatible checkpoint folder), and checkpoint/output-folder path resolution
(``_output_folder``, ``_find_latest_checkpoint``, ``_find_existing_checkpoint``).

Non-obvious rationale: split out of ``core.py`` (see that module's docstring)
because it's a distinct concern from generic env/subprocess infrastructure --
telling runs apart is a different job from loading ``.env`` -- and, unlike
``core.py``'s contents, it's only ever used by two of the four stage scripts.
Keeping it in its own module makes that usage boundary structural rather than
a comment inside a shared file that's easy to silently cross later.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from core import PROJECT_ROOT

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
