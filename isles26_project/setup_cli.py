#!/usr/bin/env python3
"""Setup stage: create .env/workspace folders and validate the environment is ready.

Expected input layout: none required for ``doctor``; ``init`` optionally takes
``--raw-root`` pointing at the raw ATLAS dataset.

What it produces: ``init`` writes ``.env`` and creates the workspace folders
listed in ``core.ensure_workspace``. ``doctor`` writes nothing -- it only
reports pass/fail status for Python version, nnU-Net storage folders, required
commands on PATH, nnunetv2/PyTorch versions, the raw dataset root, and custom
trainer discovery.

Non-obvious rationale: split out of isles26.py (see core.py's docstring) so
this stage is runnable on its own (``python setup_cli.py doctor``) as well as
via the ``isles26.py`` umbrella (``python isles26.py doctor``) -- both call
the exact same ``register()``-added subcommands, so behavior never drifts
between the two entry points.

Usage:
    python setup_cli.py init --raw-root /path/to/ATLAS3_Training_Raw
    python setup_cli.py doctor --create-dirs
"""
from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

from core import PROJECT_ROOT, build_environment, ensure_workspace, run_cli, write_env_file

EXPECTED_NNUNET_VERSION = "2.8.1"


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


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Create .env and local workspace folders")
    p.add_argument("--raw-root", help="Raw ATLAS R2.1 dataset root")
    p.add_argument("--force", action="store_true", help="Replace an existing .env")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", help="Validate dependencies, paths, and custom trainer discovery")
    p.add_argument("--create-dirs", action="store_true", help="Create missing nnU-Net storage folders")
    p.add_argument("--require-raw", action="store_true", help="Treat a missing raw dataset root as blocking")
    p.set_defaults(func=cmd_doctor)


def build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Setup stage: init and doctor")
    sub = parser.add_subparsers(dest="command", required=True)
    register(sub)
    return parser


if __name__ == "__main__":
    raise SystemExit(run_cli(build_standalone_parser()))
