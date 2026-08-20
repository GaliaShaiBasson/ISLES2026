"""Generic environment/subprocess infrastructure shared by every isles26 stage CLI
(setup_cli.py, data_prep_cli.py, train_cli.py, evaluate_cli.py) and the isles26.py
umbrella that wires them together.

Expected input layout: none directly -- this module has no CLI of its own. It's
imported by every stage script, each a sibling file in the project root.

What it produces: nothing on its own. Provides .env/environment loading
(``build_environment``), subprocess execution helpers (``run_command``), and
workspace/dataset-folder path helpers.

Non-obvious rationale: this module exists because ``isles26.py`` used to be a
single ~1,300-line file mixing this shared infrastructure with five distinct
command groups (setup, data prep, train, evaluate/report). Splitting those
groups into independently-runnable stage scripts (so each can also be driven
from its own .sh) would have meant duplicating this infrastructure -- exactly
the kind of duplicated-script drift the project's "Foundational hardening
pass" (see CLAUDE.md) already fixed once. Pulling it out here instead means
every stage script imports the same environment code rather than
re-implementing it. ``run_cli`` is the one further piece of shared boilerplate
(argparse dispatch + exception handling) so every stage script's
``if __name__ == "__main__":`` block is one line.

Deliberately does NOT contain the run-identity fingerprinting system
(``compute_run_fingerprint`` and friends) even though it's also shared
infrastructure -- that lives in ``run_identity.py`` instead, since it's only
ever used by ``train_cli.py``/``evaluate_cli.py`` (the two stages where "is
this the same run as before" matters), never by ``setup_cli.py``/
``data_prep_cli.py``. Keeping it separate makes that boundary structural
(those two stages simply have no import path to it) rather than a comment
someone could silently cross.
"""
from __future__ import annotations

import argparse
import os
import shlex
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
    "ISLES26_DATASET_ID": "2",
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
