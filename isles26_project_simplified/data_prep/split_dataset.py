#!/usr/bin/env python3
"""Subject-level train/val/test splitter for the ATLAS raw tree.

Produces four disjoint subsets:

- ``train`` / ``val``    -- used for the actual nnU-Net training/model-selection
                             (single fixed split, not 5-fold -- see PROJECT_PLAN.md
                             time-budget notes).
- ``test_id``             -- held-out subjects from sites that ARE also present in
                             train/val ("in-distribution" test).
- ``test_ood``            -- subjects from sites entirely excluded from
                             train/val/test_id ("out-of-distribution" test). This is
                             the split that actually measures cross-center
                             generalization, which is the stated motivation for
                             ISLES'26 (see ISLES2026_challenge.md).

Stratification is by lesion-size bin (reusing metadata_utils.assign_size_bin, the
same tertile bins used by prepare/sampling) and, for the OOD carve-out, by site.

Every raw file (T1w + mask) is integrity-checked (gzip-decodable) before a case is
considered usable. Cases that fail are written to a separate broken-cases report
instead of silently being dropped or silently included -- rerun this script after
a data upload finishes to confirm the broken list has cleared.

Usage:
    python split_dataset.py --raw-root /path/to/ATLAS3_Training_Raw --out-dir workspace/splits
"""
from __future__ import annotations

import argparse
import gzip
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metadata_utils import assign_size_bin, lesion_volume_mm3  # noqa: E402
from prepare_isles26_dataset import find_cases  # noqa: E402

# Sites this large are never put in the OOD carve-out: removing one wholesale
# would blow past the target OOD fraction and starve train of diversity.
OOD_INELIGIBLE_TOP_N_SITES = 4
MIN_OOD_SITES = 5
MAX_OOD_SITES = 8
OOD_SEARCH_TRIALS = 20_000

# Locked once against the complete ATLAS R3.0 raw upload (2026-08-17, 1284/1284
# usable, 0 broken) -- see CLAUDE.md decisions log. Fixed rather than re-derived so
# every subsequent experiment is measured against the same held-out sites; pass
# --ood-sites to override (e.g. after a genuine data refresh that changes the pool).
LOCKED_OOD_SITES = ["R005", "R008", "R027", "R029", "R042", "R070"]


def _is_readable_gzip(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
        return True
    except Exception:
        return False


@dataclass
class CaseRecord:
    case_id: str
    center: str
    t1w_path: Path
    mask_path: Path
    t1w_ok: bool = False
    mask_ok: bool = False
    lesion_volume_mm3: float | None = None
    size_bin: str | None = None
    broken_reason: str = field(default="")


def discover_and_validate(raw_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (usable_df, broken_df)."""
    cases = find_cases(raw_root)
    records: list[CaseRecord] = []
    for case_id, info in cases.items():
        t1w_path = info["images"][0]
        mask_path = info["label"]
        rec = CaseRecord(case_id=case_id, center=info["center"], t1w_path=t1w_path, mask_path=mask_path)
        rec.mask_ok = _is_readable_gzip(mask_path)
        rec.t1w_ok = _is_readable_gzip(t1w_path)

        reasons = []
        if not rec.mask_ok:
            reasons.append("mask unreadable/corrupt")
        if not rec.t1w_ok:
            reasons.append("T1w unreadable/corrupt (likely still uploading)")
        rec.broken_reason = "; ".join(reasons)

        if rec.mask_ok:
            try:
                rec.lesion_volume_mm3 = lesion_volume_mm3(mask_path)
            except Exception as exc:  # defensive: gzip was fine but NIfTI parse failed
                rec.mask_ok = False
                rec.broken_reason = (rec.broken_reason + f"; mask parse failed: {exc}").strip("; ")

        records.append(rec)

    df = pd.DataFrame([r.__dict__ for r in records])
    usable_mask = df["mask_ok"] & df["t1w_ok"]
    usable = df[usable_mask].copy()
    broken = df[~usable_mask].copy()

    if not usable.empty:
        usable["size_bin"] = assign_size_bin(usable["lesion_volume_mm3"])
        usable.loc[usable["lesion_volume_mm3"] == 0, "size_bin"] = "empty"

    return usable, broken


def _bin_proportions(df: pd.DataFrame) -> dict[str, float]:
    counts = df["size_bin"].value_counts(normalize=True)
    return counts.to_dict()


def choose_ood_sites(usable: pd.DataFrame, target_frac: float, seed: int) -> list[str]:
    site_counts = usable.groupby("center").size().sort_values(ascending=False)
    ineligible = set(site_counts.index[:OOD_INELIGIBLE_TOP_N_SITES])
    eligible_sites = [s for s in site_counts.index if s not in ineligible]

    overall_props = _bin_proportions(usable)
    target_n = target_frac * len(usable)

    rng = random.Random(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(OOD_SEARCH_TRIALS):
        k = rng.randint(MIN_OOD_SITES, MAX_OOD_SITES)
        if k > len(eligible_sites):
            continue
        candidate = rng.sample(eligible_sites, k)
        subset = usable[usable["center"].isin(candidate)]
        n = len(subset)
        if n == 0:
            continue
        size_penalty = abs(n - target_n) / target_n
        cand_props = _bin_proportions(subset)
        bin_penalty = sum(abs(cand_props.get(b, 0.0) - overall_props.get(b, 0.0)) for b in overall_props)
        score = size_penalty + bin_penalty
        if best is None or score < best[0]:
            best = (score, candidate)

    assert best is not None, "OOD site search failed to find any valid candidate"
    return sorted(best[1])


def stratified_split(
    usable: pd.DataFrame,
    ood_sites: list[str],
    train_frac: float,
    val_frac: float,
    test_id_frac: float,
    seed: int,
) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    df = usable.copy()
    df["split"] = ""
    df.loc[df["center"].isin(ood_sites), "split"] = "test_ood"

    pool = df[df["split"] == ""].copy()
    # "empty" (zero-lesion) cases are too few dataset-wide (single digits) to survive
    # as their own class through *two* successive stratified splits -- sklearn needs
    # >=2 members per class per split, and a second split can starve a class that had
    # only 1-2 members left after the first. For split *mechanics* only, fold "empty"
    # into "small" (volume 0 is the extreme end of "small") so they ride along with a
    # large class and land wherever that class's split naturally puts them -- mostly
    # train, occasionally test, matching the intended handling without hand-picking
    # specific cases. The real "empty" label is preserved in size_bin/the manifest.
    strat_bin = pool["size_bin"].replace("empty", "small")
    #
    # train_frac/val_frac/test_id_frac are fractions of the *whole* usable set, but
    # the OOD carve-out already removed a site-driven (not exactly test_ood_frac-sized)
    # chunk from the pool. Re-normalize the three remaining fractions to ratios that
    # sum to 1 *within this pool*, rather than assuming they already do globally.
    pool_frac_total = train_frac + val_frac + test_id_frac
    assert pool_frac_total > 0, "train_frac + val_frac + test_id_frac must be > 0"

    pool_train, pool_rest = train_test_split(
        pool, test_size=(val_frac + test_id_frac) / pool_frac_total, stratify=strat_bin, random_state=seed
    )
    strat_bin_rest = strat_bin.loc[pool_rest.index]
    rest_test_size = test_id_frac / (val_frac + test_id_frac)
    pool_val, pool_test_id = train_test_split(
        pool_rest, test_size=rest_test_size, stratify=strat_bin_rest, random_state=seed
    )

    df.loc[pool_train.index, "split"] = "train"
    df.loc[pool_val.index, "split"] = "val"
    df.loc[pool_test_id.index, "split"] = "test_id"

    assert (df["split"] != "").all(), "every usable case must be assigned to a split"
    return df


def subsample_per_split(split_df: pd.DataFrame, n_per_split: int, seed: int) -> pd.DataFrame:
    """Shrink an already-finalized split down to ~n_per_split cases per split, for a fast

    full-pipeline smoke test (prepare -> preprocess -> train debug -> evaluate -> aggregate ->
    plot) instead of running it against the real ~1300-case dataset. Does NOT re-derive the
    split (train/val/test_id/test_ood membership and OOD sites are unchanged) -- just takes a
    stratified-by-size_bin random subsample within each split, so the same properties the real
    split was designed for (balanced bins, real OOD sites) are preserved at a smaller scale.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for split_name, group in split_df.groupby("split", observed=True):
        n = min(n_per_split, len(group))
        # Sample proportionally within each size_bin so tiny bins (e.g. "empty") aren't
        # either dropped entirely or over-represented relative to the real split. Plain
        # per-bin loop rather than groupby.apply -- simpler than working around apply's
        # grouping-column-inclusion semantics for no benefit here.
        bin_parts = []
        for _, bin_group in group.groupby("size_bin", observed=True):
            n_this_bin = min(len(bin_group), max(1, round(n * len(bin_group) / len(group))))
            bin_parts.append(bin_group.sample(n=n_this_bin, random_state=rng.randint(0, 2**31 - 1)))
        sampled = pd.concat(bin_parts, ignore_index=True)
        if len(sampled) > n:
            sampled = sampled.sample(n=n, random_state=rng.randint(0, 2**31 - 1))
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--test-id-frac", type=float, default=0.10)
    parser.add_argument("--test-ood-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ood-sites",
        default=None,
        help=(
            "Comma-separated site IDs to use as the fixed OOD test set, overriding "
            f"LOCKED_OOD_SITES ({','.join(LOCKED_OOD_SITES)}). Pass 'search' to run the "
            "randomized site-selection search instead (used before the OOD set was locked)."
        ),
    )
    parser.add_argument(
        "--sample-per-split",
        type=int,
        default=None,
        help=(
            "If set, shrink each split to ~N cases (stratified by size_bin) after computing "
            "the real split, for a fast full-pipeline smoke test. Use a different --out-dir "
            "than the real split (e.g. workspace/splits_sample) to avoid overwriting it."
        ),
    )
    args = parser.parse_args()

    frac_total = args.train_frac + args.val_frac + args.test_id_frac + args.test_ood_frac
    if abs(frac_total - 1.0) > 1e-6:
        parser.error(f"--train/val/test-id/test-ood fracs must sum to 1.0 (got {frac_total})")

    raw_root = Path(args.raw_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning and validating raw tree under {raw_root} ...")
    usable, broken = discover_and_validate(raw_root)
    print(f"Usable cases (both T1w and mask readable): {len(usable)}")
    print(f"Broken cases (excluded): {len(broken)}")

    broken_path = out_dir / "broken_cases.csv"
    broken.to_csv(broken_path, index=False)
    print(f"Wrote broken-case report: {broken_path}")

    n_t1w_broken = broken["t1w_ok"].eq(False).sum()
    total_scanned = len(usable) + len(broken)
    upload_incomplete = total_scanned == 0 or (n_t1w_broken / total_scanned) > 0.05
    if upload_incomplete:
        print(
            "\n[warning] a large fraction of T1w scans are unreadable. If the raw-data "
            "upload/transfer is still in progress, re-run this script once it's finished "
            "-- do not treat the split below as final while this warning is showing."
        )

    if usable.empty:
        print("No usable cases yet; nothing to split.")
        return 1

    print("\nSize-bin distribution among usable cases:")
    print(usable["size_bin"].value_counts(dropna=False).to_string())

    if args.ood_sites == "search":
        ood_sites = choose_ood_sites(usable, target_frac=args.test_ood_frac, seed=args.seed)
        print(f"\n[search] selected {len(ood_sites)} OOD (held-out) sites: {ood_sites}")
    else:
        ood_sites = sorted(args.ood_sites.split(",")) if args.ood_sites else sorted(LOCKED_OOD_SITES)
        unknown = set(ood_sites) - set(usable["center"].unique())
        if unknown:
            print(f"[warning] requested OOD sites not present in usable data: {sorted(unknown)}")
        print(f"\nUsing fixed OOD (held-out) sites: {ood_sites}")

    split_df = stratified_split(
        usable,
        ood_sites=ood_sites,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_id_frac=args.test_id_frac,
        seed=args.seed,
    )

    # Leakage assertion: every case in exactly one split.
    assert split_df["case_id"].is_unique
    assert set(split_df["split"].unique()) <= {"train", "val", "test_id", "test_ood"}

    if args.sample_per_split:
        split_df = subsample_per_split(split_df, args.sample_per_split, args.seed)
        print(f"\n[sample] shrunk to ~{args.sample_per_split} cases per split -> {len(split_df)} total")

    for split_name in ("train", "val", "test_id", "test_ood"):
        subset = split_df[split_df["split"] == split_name]
        path = out_dir / f"{split_name}.csv"
        subset[["case_id", "center", "lesion_volume_mm3", "size_bin"]].to_csv(path, index=False)
        print(f"  {split_name:9s} n={len(subset):4d}  sites={subset['center'].nunique():2d}  -> {path}")

    manifest_path = out_dir / "manifest.csv"
    split_df[["case_id", "center", "lesion_volume_mm3", "size_bin", "split"]].to_csv(manifest_path, index=False)
    print(f"\nWrote combined manifest: {manifest_path}")

    print("\nPer-split size-bin composition (fractions):")
    comp = pd.crosstab(split_df["split"], split_df["size_bin"], normalize="index")
    print(comp.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
