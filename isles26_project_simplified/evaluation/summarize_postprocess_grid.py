#!/usr/bin/env python3
"""Build grid_summary_val.csv + selected_combo.json from a postprocess grid's
per-combo results_<tag>.csv files -- factored out of run_postprocess_grid.sh's
original inline Python heredoc (see git history) so the selection logic is
independently testable/rerunnable without re-running the whole grid, and so
evaluation/run_postprocess_grid.sh (the generalized, any-trainer version) and
the original baseline-500-specific script can share one implementation.

Expected input: `--val-root` containing `*_config.json` (min_voxels/
connectivity/tag per combo, written by the grid orchestrator) and matching
`results_<tag>.csv` (per-case metrics from compute_metrics.py).

What it produces (under `--out-root`):
- `grid_summary_val.csv` -- one row per combo, sorted by (connectivity, min_voxels).
- `selected_combo.json` -- the auto-selected winner + the selection rule used,
  for `run_postprocess_grid.sh`'s phase 2 (or a caller's own use) to read back.

Selection rule (default, matches the original baseline-500 grid's precedent):
lowest val `hd95_mean`, `dice_mean` as tiebreaker -- HD95 is the metric small
spurious components distort most (see CLAUDE.md), so it's the most informative
single criterion for connected-component filtering specifically. Override via
`--selection-metric`/`--tiebreak-metric` if a different rule is deliberately
wanted for a given call -- not silently, since changing this changes what
"winner" means.

Usage:
    python evaluation/summarize_postprocess_grid.py --val-root workspace/postprocess_grid/wideaug500/val_search --out-root workspace/postprocess_grid/wideaug500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-root", required=True, type=Path)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--selection-metric", default="hd95_mean")
    ap.add_argument("--selection-lower-is-better", action="store_true", default=True)
    ap.add_argument("--tiebreak-metric", default="dice_mean")
    args = ap.parse_args()

    rows = []
    for config_json in sorted(args.val_root.glob("*_config.json")):
        cfg = json.loads(config_json.read_text())
        tag = cfg["tag"]
        results_csv = args.val_root / f"results_{tag}.csv"
        pp_summary = args.val_root / f"{tag}_predTs" / "postprocess_summary.csv"
        if not results_csv.is_file():
            print(f"[missing] {results_csv} -- {tag} did not complete, excluded from summary")
            continue
        df = pd.read_csv(results_csv)
        row = {
            "tag": tag, "min_voxels": cfg["min_voxels"], "connectivity": cfg["connectivity"],
            "n_cases": len(df), "dice_mean": df["dice"].mean(), "dice_median": df["dice"].median(),
            "hd95_mean": df["hd95_mm"].mean(), "hd95_median": df["hd95_mm"].median(),
        }
        if "lesion_f1" in df.columns:
            row["lesion_f1_mean"] = df["lesion_f1"].mean()
        if pp_summary.is_file():
            pp = pd.read_csv(pp_summary)
            row["cases_with_component_removed"] = int((pp["components_removed"] > 0).sum())
            row["total_voxels_removed"] = int((pp["voxels_before"] - pp["voxels_after"]).sum())
        rows.append(row)

    if not rows:
        print("[FATAL] no combo produced results -- nothing to summarize")
        return 1

    summary = pd.DataFrame(rows).round(4).sort_values(["connectivity", "min_voxels"])
    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_root / "grid_summary_val.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nWrote {summary_csv}\n")
    print(summary.to_string(index=False))

    no_op_rows = summary[summary.min_voxels == 1]
    raw = no_op_rows.iloc[0] if not no_op_rows.empty else None

    ascending = args.selection_lower_is_better
    best = summary.sort_values(
        [args.selection_metric, args.tiebreak_metric], ascending=[ascending, not ascending]
    ).iloc[0]
    if raw is not None:
        print(f"\nRaw (min_voxels=1) control: dice={raw.dice_mean:.4f} hd95={raw.hd95_mean:.4f}")
    print(f"Selected winner (lowest val {args.selection_metric}, {args.tiebreak_metric} as tiebreaker): "
          f"tag={best.tag} min_voxels={best.min_voxels:.0f} connectivity={best.connectivity:.0f} "
          f"dice={best.dice_mean:.4f} hd95={best.hd95_mean:.4f}")

    selected = {
        "tag": best.tag, "min_voxels": int(best.min_voxels), "connectivity": int(best.connectivity),
        "selection_rule": f"lowest val {args.selection_metric}, {args.tiebreak_metric} as tiebreaker",
        "val_dice_mean": float(best.dice_mean), "val_hd95_mean": float(best.hd95_mean),
    }
    if raw is not None:
        selected["val_dice_mean_raw_control"] = float(raw.dice_mean)
        selected["val_hd95_mean_raw_control"] = float(raw.hd95_mean)
    (args.out_root / "selected_combo.json").write_text(json.dumps(selected, indent=2) + "\n")
    print(f"Wrote {args.out_root / 'selected_combo.json'} -- auto-selected candidate, REVIEW before using in the report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
