#!/usr/bin/env python
"""
Copies the custom loss + trainer files in this folder into your installed
`nnunetv2` package, under `training/nnUNetTrainer/variants/isles26_ext/`.

This follows nnU-Net's own convention: any `nnUNetTrainer` subclass that is
importable from *anywhere inside* the `nnunetv2` package is auto-discovered
by class name when you pass `-tr <ClassName>` to `nnUNetv2_train`. Placing
custom code inside the installed package (rather than trying to get nnU-Net
to import from an arbitrary external path) is the standard, low-friction way
the nnU-Net community extends trainers.

Run this once after `pip install nnunetv2`, and again any time you edit the
files in this folder:

    python install_custom_trainers.py
"""
import shutil
import sys
from pathlib import Path


def main():
    try:
        import nnunetv2
    except ImportError:
        sys.exit("nnunetv2 is not importable in this Python environment. "
                  "Run `pip install nnunetv2` first (see requirements.txt), "
                  "then re-run this script in the same environment.")

    pkg_root = Path(nnunetv2.__file__).parent
    target_dir = pkg_root / "training" / "nnUNetTrainer" / "variants" / "isles26_ext"
    target_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(__file__).parent
    files_to_copy = [
        "losses.py",
        "nnUNetTrainerLossVariants.py",
        "nnUNetTrainerLesionAwareSampling.py",
        "nnUNetTrainerDebugFast.py",
    ]

    for fname in files_to_copy:
        src = src_dir / fname
        dst = target_dir / fname
        shutil.copy(src, dst)
        print(f"Copied {src} -> {dst}")

    # `nnUNetTrainerLossVariants.py` does `from losses import ...` (relative
    # to this project folder during development). Once copied next to
    # `losses.py` inside the package, that import still resolves correctly
    # because both files now live in the same directory
    # (`variants/isles26_ext/`), which Python adds implicitly via nnU-Net's
    # own recursive class discovery (it imports each module by file path).

    init_file = target_dir / "__init__.py"
    init_file.touch(exist_ok=True)

    print(f"\nDone. Custom trainers installed under: {target_dir}")
    print("Verify with: nnUNetv2_train -h   (new -tr options should be accepted, "
          "though they won't be listed explicitly -- nnU-Net will simply not error "
          "on the class name if discovery worked).")
    print("Quick check: python -c \"from nnunetv2.training.nnUNetTrainer.variants.isles26_ext.nnUNetTrainerLossVariants import nnUNetTrainerFocalTversky; print('OK')\"")


if __name__ == "__main__":
    main()
