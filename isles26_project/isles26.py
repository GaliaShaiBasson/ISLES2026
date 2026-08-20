#!/usr/bin/env python3
"""Cross-platform command runner for the ISLES/ATLAS nnU-Net project.

Run ``python isles26.py --help`` for the available commands.
The runner loads project settings from ``.env`` and passes them only to the
child nnU-Net processes, so users do not need to configure permanent shell
variables or run platform-specific scripts.

Non-obvious rationale: this used to be a single ~1,300-line file mixing
shared environment/fingerprinting infrastructure with five distinct command
groups (setup, data prep, train, evaluate/report). It's now a thin umbrella:
each command group lives in its own stage script (``setup_cli.py``,
``data_prep_cli.py``, ``train_cli.py``, ``evaluate_cli.py``), independently
runnable on its own (e.g. ``python train_cli.py baseline-500
--dataset-id 2``) or wrappable in its own .sh -- and this file just imports
each stage's ``register()`` function to expose all of them under one
``python isles26.py <command>`` entry point. Shared logic (env/.env loading,
run-identity fingerprinting) lives in ``core.py``, imported by every stage so
none of them duplicate it. See ``core.py``'s docstring for the full
rationale, and ``REPO_STRUCTURE_PLAN.md`` for the reorg this was part of.
"""
from __future__ import annotations

import argparse

import data_prep_cli
import evaluate_cli
import setup_cli
import train_cli
from core import run_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One cross-platform entry point for data preparation, nnU-Net training, and evaluation."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    setup_cli.register(sub)
    data_prep_cli.register(sub)
    train_cli.register(sub)
    evaluate_cli.register(sub)
    return parser


def main() -> int:
    return run_cli(build_parser())


if __name__ == "__main__":
    raise SystemExit(main())
