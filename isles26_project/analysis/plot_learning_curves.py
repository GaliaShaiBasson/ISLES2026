#!/usr/bin/env python3
"""Parse nnU-Net training_log_*.txt files for one or more runs and plot
learning curves (train/val loss, pseudo Dice) and the learning-rate schedule.

Expected input layout: nnU-Net's own ``fold_0/training_log_<timestamp>.txt``
files, each containing repeated per-epoch blocks of the form::

    Epoch <n>
    Current learning rate: <lr>
    train_loss <value>
    val_loss <value>
    Pseudo dice [np.float32(<value>)]
    Epoch time: <seconds> s

Produces: ``<out-dir>/learning_curves_<tag>.png`` (3-panel: train loss, val
loss, pseudo Dice vs. epoch), ``<out-dir>/lr_schedule_<tag>.png`` (the
PolyLR decay), ``<out-dir>/train_val_overlay_<tag>.png`` (one small-multiple
subplot per trainer, train loss vs. val loss overlaid on the same axes, to
read the train/val gap per condition at a glance), and one
``<out-dir>/learning_curve_<run-label>.csv`` per run (epoch, lr, train_loss,
val_loss, pseudo_dice) so the parsed data doesn't need re-parsing to check
exact numbers.

Two ways to select which runs to plot:
  - ``--log 'label=path'`` (repeatable): explicit, manually-labeled runs.
  - Default (no ``--log``): auto-discovers every trainer in
    ``--runs-index`` (the ``runs_index.csv`` written by
    ``isles26.py aggregate``), resolves each trainer's ``fold_0`` directory
    from its ``checkpoint_final`` path, and parses whatever
    ``training_log_*.txt`` files are in there. Narrow with ``--trainer``
    (repeatable exact trainer names) if only a subset is wanted.

Non-obvious rationale:
  - nnU-Net logs the learning rate as plain text that can appear in
    scientific notation near the end of a PolyLR decay (e.g. ``7e-05``) --
    a naive ``[\\d.]+`` regex silently truncates that to ``7``, producing a
    false spike at the end of the LR curve. The regex here explicitly
    handles the ``e[+-]NN`` suffix.
  - A trainer's ``fold_0`` directory can hold more than one
    ``training_log_*.txt`` (nnU-Net starts a fresh log file on every
    process restart, including a resume or a later validate-only/predict
    invocation that reopens the same folder). All matching logs are parsed
    and merged by epoch number (later files win on overlap) so the curve
    covers the full training history, not just the last invocation.

Usage:
    # explicit runs: baseline at 250 vs. 500 vs. (once it exists) 1000 epochs
    python analysis/plot_learning_curves.py \\
        --log "250 epochs=/path/to/fold_0/training_log_2026_8_17_02_13_04.txt" \\
        --log "500 epochs=/path/to/fold_0/training_log_2026_8_18_01_35_16.txt" \\
        # --log "1000 epochs=/path/to/fold_0/training_log_<timestamp>.txt" \\  # not run yet
        --tag baseline_250_vs_500_vs_1000 --out-dir figures

    # auto-discover every trainer from the current runs_index.csv
    python analysis/plot_learning_curves.py --tag all_trainers --out-dir figures
"""
import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt

NUM = r"[\-\d.]+(?:[eE][\-+]?\d+)?"  # plain or scientific notation, e.g. 7e-05
EPOCH_RE = re.compile(r"Epoch (\d+)")
LR_RE = re.compile(rf"Current learning rate: ({NUM})")
TRAIN_LOSS_RE = re.compile(rf"train_loss ({NUM})")
VAL_LOSS_RE = re.compile(rf"val_loss ({NUM})")
DICE_RE = re.compile(rf"Pseudo dice \[np\.float32\(({NUM})\)\]")

COLOR_CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
DEFAULT_RUNS_INDEX = os.path.join("workspace", "evaluation", "runs_index.csv")


def parse_log(path):
    """Parse one training_log_*.txt into per-epoch lr/train_loss/val_loss/dice."""
    epochs, lrs, train_losses, val_losses, dices = [], [], [], [], []
    cur_epoch = cur_lr = train_loss = val_loss = None
    with open(path) as f:
        for line in f:
            m = EPOCH_RE.search(line)
            if m and "Epoch time" not in line:
                cur_epoch = int(m.group(1))
                continue
            m = LR_RE.search(line)
            if m:
                cur_lr = float(m.group(1))
                continue
            m = TRAIN_LOSS_RE.search(line)
            if m and cur_epoch is not None:
                train_loss = float(m.group(1))
                continue
            m = VAL_LOSS_RE.search(line)
            if m and cur_epoch is not None:
                val_loss = float(m.group(1))
                continue
            m = DICE_RE.search(line)
            if m and cur_epoch is not None:
                # Pseudo dice is the last field logged per epoch -> commit the row.
                epochs.append(cur_epoch)
                lrs.append(cur_lr)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                dices.append(float(m.group(1)))
    return {
        "epoch": epochs,
        "lr": lrs,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "pseudo_dice": dices,
    }


def parse_logs_merged(paths):
    """Parse and merge multiple training_log_*.txt files from the same fold_0
    dir into one continuous per-epoch series (later files, by filename sort,
    win on an overlapping epoch -- see module docstring)."""
    by_epoch = {}
    for path in sorted(paths):
        d = parse_log(path)
        for i, epoch in enumerate(d["epoch"]):
            by_epoch[epoch] = (d["lr"][i], d["train_loss"][i], d["val_loss"][i], d["pseudo_dice"][i])
    epochs = sorted(by_epoch)
    return {
        "epoch": epochs,
        "lr": [by_epoch[e][0] for e in epochs],
        "train_loss": [by_epoch[e][1] for e in epochs],
        "val_loss": [by_epoch[e][2] for e in epochs],
        "pseudo_dice": [by_epoch[e][3] for e in epochs],
    }


def discover_runs_from_index(runs_index_path, trainers=None):
    """Auto-discover {trainer_name: parsed_data} from a runs_index.csv (as
    written by isles26.py aggregate): one entry per unique trainer, its
    training log(s) found next to its checkpoint_final."""
    if not os.path.exists(runs_index_path):
        raise SystemExit(
            f"--runs-index not found: {runs_index_path!r}. Pass --log explicitly, or --runs-index "
            "pointing at the CSV written by 'isles26.py aggregate'."
        )
    fold_dirs = {}  # trainer -> fold_0 dir
    with open(runs_index_path, newline="") as f:
        for row in csv.DictReader(f):
            trainer = row.get("trainer")
            checkpoint = row.get("checkpoint_final")
            if not trainer or not checkpoint:
                continue
            if trainers and trainer not in trainers:
                continue
            fold_dirs.setdefault(trainer, os.path.dirname(checkpoint))

    if trainers:
        missing = set(trainers) - set(fold_dirs)
        if missing:
            raise SystemExit(f"--trainer name(s) not found in {runs_index_path}: {sorted(missing)}")

    runs = {}
    for trainer, fold_dir in sorted(fold_dirs.items()):
        log_paths = glob.glob(os.path.join(fold_dir, "training_log_*.txt"))
        if not log_paths:
            print(f"[warn] no training_log_*.txt found for {trainer} in {fold_dir}, skipping")
            continue
        runs[trainer] = parse_logs_merged(log_paths)
    return runs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--log", action="append", dest="logs",
        help="Run label and log path as 'label=path', e.g. '500 epochs=/path/training_log_*.txt'. "
             "Repeat --log for each run to compare. If omitted, runs are auto-discovered from --runs-index "
             "(every trainer there, or just --trainer names if given).",
    )
    p.add_argument(
        "--runs-index", default=DEFAULT_RUNS_INDEX,
        help=f"runs_index.csv to auto-discover trainers from when --log is omitted (default: {DEFAULT_RUNS_INDEX}).",
    )
    p.add_argument(
        "--trainer", action="append", dest="trainers",
        help="Restrict auto-discovery to this trainer name (repeatable, exact match against the 'trainer' "
             "column of --runs-index). Ignored if --log is given.",
    )
    p.add_argument("--tag", required=True, help="Suffix used in output filenames, e.g. 'baseline_250_vs_500'.")
    p.add_argument("--out-dir", default="figures", help="Directory to write PNGs/CSVs into (default: figures).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.logs:
        runs = {}
        for spec in args.logs:
            label, _, path = spec.partition("=")
            if not path:
                raise SystemExit(f"--log must be 'label=path', got: {spec!r}")
            runs[label] = parse_log(path)
    else:
        runs = discover_runs_from_index(args.runs_index, trainers=args.trainers)
        if not runs:
            raise SystemExit(f"no trainers with parseable logs found via {args.runs_index}")

    # tab20 cycles through 20 distinct colors before repeating -- the fixed 5-color
    # COLOR_CYCLE is fine for a hand-picked --log comparison but auto-discovery can
    # easily pull in a dozen+ trainers.
    palette = COLOR_CYCLE if len(runs) <= len(COLOR_CYCLE) else [
        plt.cm.tab20(i % 20) for i in range(len(runs))
    ]
    colors = dict(zip(runs.keys(), palette))
    for label, d in runs.items():
        n = len(d["epoch"])
        final_dice = d["pseudo_dice"][-1] if d["pseudo_dice"] else float("nan")
        print(f"{label}: parsed {n} epochs (final pseudo dice {final_dice:.4f})")

    # --- Learning curves: train loss, val loss, pseudo dice ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 11))
    panels = [
        ("train_loss", "Train loss", "training loss vs epoch"),
        ("val_loss", "Validation loss", "validation loss vs epoch"),
        ("pseudo_dice", "Pseudo Dice (EMA val)", "pseudo Dice vs epoch"),
    ]
    for ax, (key, ylabel, title) in zip(axes, panels):
        for label, d in runs.items():
            ax.plot(d["epoch"], d[key], label=label, color=colors[label], lw=1.2)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Baseline: {title}")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    out1 = os.path.join(args.out_dir, f"learning_curves_{args.tag}.png")
    fig.savefig(out1, dpi=150)
    print(f"saved {out1}")

    # --- Learning rate schedule ---
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    for label, d in runs.items():
        ax2.plot(d["epoch"], d["lr"], label=label, color=colors[label], lw=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning rate")
    ax2.set_title("Baseline: learning-rate schedule (PolyLR)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(args.out_dir, f"lr_schedule_{args.tag}.png")
    fig2.savefig(out2, dpi=150)
    print(f"saved {out2}")

    # --- Train vs val loss overlay, one small-multiple subplot per trainer ---
    # Same axes per trainer (not separate train/val panels) so the train/val gap --
    # the overfitting signal -- reads directly off the vertical distance between the
    # two lines, instead of having to eyeball two different panels against each other.
    labels = list(runs.keys())
    ncols = min(3, len(labels)) or 1
    nrows = -(-len(labels) // ncols)  # ceil
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    for ax, label in zip(axes3.flat, labels):
        d = runs[label]
        ax.plot(d["epoch"], d["train_loss"], label="train_loss", color="#1f77b4", lw=1.2)
        ax.plot(d["epoch"], d["val_loss"], label="val_loss", color="#d62728", lw=1.2)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes3.flat[len(labels):]:
        ax.set_visible(False)
    fig3.suptitle("Train vs. validation loss per trainer")
    fig3.tight_layout()
    out3 = os.path.join(args.out_dir, f"train_val_overlay_{args.tag}.png")
    fig3.savefig(out3, dpi=150)
    print(f"saved {out3}")

    # --- dump parsed data to CSV for reproducibility ---
    for label, d in runs.items():
        safe_label = re.sub(r"\W+", "_", label).strip("_")
        out_csv = os.path.join(args.out_dir, f"learning_curve_{safe_label}.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "lr", "train_loss", "val_loss", "pseudo_dice"])
            w.writerows(zip(d["epoch"], d["lr"], d["train_loss"], d["val_loss"], d["pseudo_dice"]))
        print(f"saved {out_csv}")


if __name__ == "__main__":
    main()
