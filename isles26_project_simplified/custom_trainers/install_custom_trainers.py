#!/usr/bin/env python3
"""Compatibility notice for the old installer.

nnU-Net 2.8.1 supports external trainer directories through
``nnUNet_extTrainer``. The project runner configures that automatically, so
copying files into site-packages is no longer necessary or recommended.
"""
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    print("No installation is required.")
    print(f"Run commands through: python {project_root / 'isles26.py'} ...")
    print(f"The runner exposes custom trainers from: {project_root}")


if __name__ == "__main__":
    main()
