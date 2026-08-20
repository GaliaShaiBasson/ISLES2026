#!/usr/bin/env python3
"""Subject-level train/val/test/final-holdout splitter for the ATLAS raw tree.

Produces six disjoint subsets, in two tiers:

Dev-time (looked at repeatedly during the project):
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

Hidden (carved out first, never looked at until the project's final evaluation --
this project's stand-in for the real ISLES'26 challenge test set, which we never
actually received):
- ``final_holdout_id``    -- a stratified subject-level sample from the same site
                             pool as train/val/test_id/test_ood.
- ``final_holdout_ood``   -- entire sites never used anywhere else in the project,
                             deliberately including any site not present in the
                             previous (locked, 1,284-case) split at all -- e.g. the
                             ``SOOP`` site, discovered 2026-08-17 (see CLAUDE.md) --
                             plus additional sites chosen the same way test_ood's
                             sites were, for multi-center diversity in the hidden
                             check rather than one site's scanner signature alone.

Stratification is by lesion-size bin (reusing metadata_utils.assign_size_bin, the
same tertile bins used by prepare/sampling) and, for both OOD carve-outs, by site.

Every raw file (T1w + mask) is integrity-checked (gzip-decodable) before a case is
considered usable. Cases that fail are written to a separate broken-cases report
instead of silently being dropped or silently included -- rerun this script after
a data upload finishes to confirm the broken list has cleared.

The two final_holdout_*.csv files (and the corresponding image/ground-truth copies
this script writes under --final-holdout-dir) are deliberately NOT meant to be
committed or inspected -- see the --final-holdout-dir docstring below and the
repo .gitignore. manifest.csv only carries the four dev-time splits.

Usage:
    python split_dataset.py --raw-root /path/to/ATLAS3_Training_Raw --out-dir workspace/splits_dataset002
"""
from __future__ import annotations

import argparse
import gzip
import random
import shutil
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

# Locked once against what was, at the time, believed to be the complete ATLAS R3.0
# raw upload (2026-08-17, 1284/1284 usable, 0 broken) -- see CLAUDE.md decisions log.
# Fixed rather than re-derived so every subsequent experiment is measured against the
# same held-out sites; pass --ood-sites to override (e.g. after a genuine data
# refresh that changes the pool). Turned out NOT to be the complete upload -- the
# ``SOOP`` site (169 cases) was present on disk the whole time but silently missed by
# a naming-pattern bug in find_cases() (fixed 2026-08-17, see prepare_isles26_dataset.py).
# These sites remain reserved for dev-time test_ood specifically (never eligible for
# the final_holdout_ood carve-out below) so the existing dev-time comparisons stay
# interpretable even as the full 1,453-case pool is split.
LOCKED_OOD_SITES = ["R005", "R008", "R027", "R029", "R042", "R070"]

# Sites guaranteed to be in the hidden final_holdout_ood carve-out: any site with
# zero presence in the previous (1,284-case) split is, by construction, a site no
# experiment has ever trained or dev-evaluated on -- the best available stand-in for
# a genuinely unseen challenge-test-set site. Additional sites are added by search
# (see choose_site_pool) purely for multi-center diversity in the hidden check.
FINAL_HOLDOUT_OOD_GUARANTEED_SITES = ["SOOP"]
FINAL_HOLDOUT_OOD_TARGET_FRAC = 0.10
FINAL_HOLDOUT_ID_FRAC = 0.05


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
    return choose_site_pool(usable, target_frac=target_frac, seed=seed)


def choose_site_pool(
    usable: pd.DataFrame,
    target_frac: float,
    seed: int,
    guaranteed_sites: list[str] | None = None,
    excluded_sites: list[str] | None = None,
    min_sites: int = MIN_OOD_SITES,
    max_sites: int = MAX_OOD_SITES,
) -> list[str]:
    """Randomized search (seeded) for a set of whole sites matching a target size
    and the dataset-wide lesion-size-bin proportions.

    ``guaranteed_sites`` are always included (e.g. a site with zero presence in a
    previous split, so it's certainly never been trained/evaluated on) -- the search
    only chooses *additional* sites on top of them. ``excluded_sites`` are never
    eligible (e.g. sites already reserved for a different carve-out). The largest
    ``OOD_INELIGIBLE_TOP_N_SITES`` sites are always excluded regardless, same
    rationale as the original OOD search: removing one wholesale would blow past
    the target fraction and starve the remaining pool of diversity.
    """
    guaranteed = list(guaranteed_sites or [])
    excluded = set(excluded_sites or [])

    site_counts = usable.groupby("center").size().sort_values(ascending=False)
    ineligible = set(site_counts.index[:OOD_INELIGIBLE_TOP_N_SITES]) | excluded
    eligible_sites = [s for s in site_counts.index if s not in ineligible and s not in guaranteed]

    overall_props = _bin_proportions(usable)
    target_n = target_frac * len(usable)
    guaranteed_n = len(usable[usable["center"].isin(guaranteed)])

    rng = random.Random(seed)
    best: tuple[float, list[str]] | None = None
    for _ in range(OOD_SEARCH_TRIALS):
        k = rng.randint(min_sites, max_sites)
        k_extra = max(0, k - len(guaranteed))
        if k_extra > len(eligible_sites):
            continue
        candidate = guaranteed + rng.sample(eligible_sites, k_extra)
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

    assert best is not None, "Site pool search failed to find any valid candidate"
    if guaranteed_n > target_n:
        print(
            f"[warning] guaranteed sites alone ({guaranteed_n} cases) already exceed "
            f"the target fraction ({target_n:.0f} cases) -- search picked the closest "
            "match, but consider raising target_frac."
        )
    return sorted(best[1])


def carve_final_holdout_id(
    pool: pd.DataFrame, frac: float, seed: int, excluded_sites: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified-by-size-bin subject-level sample, held out from the same site pool
    as train/val/test_id -- the "in-distribution" half of the hidden final holdout.

    ``excluded_sites`` (the dev-time OOD sites, e.g. LOCKED_OOD_SITES) must never be
    eligible for this sample: an "in-distribution" case is only meaningful if it
    comes from a site the model actually trains on. Without this, the ID carve was
    blind to which sites become test_ood and could (did, in practice -- 7/64 cases in
    the 1,453-case split) draw cases from OOD-reserved sites, silently mislabeling
    them "in-distribution" -- see CLAUDE.md. Excluded-site cases are passed straight
    through to ``remaining_pool_df`` untouched, where the caller's own OOD-site
    assignment (stratified_split) still correctly routes them to test_ood.

    Returns (final_holdout_id_df, remaining_pool_df).
    """
    from sklearn.model_selection import train_test_split

    excluded = set(excluded_sites or [])
    eligible = pool[~pool["center"].isin(excluded)].copy()
    ineligible = pool[pool["center"].isin(excluded)].copy()

    strat_bin = eligible["size_bin"].replace("empty", "small")
    remaining, holdout = train_test_split(eligible, test_size=frac, stratify=strat_bin, random_state=seed)
    if not ineligible.empty:
        remaining = pd.concat([remaining, ineligible]).sort_index()
    return holdout, remaining


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


def write_final_holdout(final_holdout_id_df: pd.DataFrame, final_holdout_ood_df: pd.DataFrame, final_holdout_dir: Path) -> None:
    """Write the hidden final holdout to its own directory tree, separate from
    --out-dir and from anything nnU-Net-visible (imagesTr/imagesTs).

    Layout::

        <final_holdout_dir>/
            final_holdout_id.csv                       # case_id/center/volume/size_bin, no images
            final_holdout_ood.csv
            images/{id,ood}/<case_id>_0000.nii.gz       # T1w only -- safe to run prediction against
            ground_truth_DO_NOT_TOUCH/{id,ood}/<case_id>.nii.gz   # masks, kept separate on purpose

    Ground truth lives under a directory named to make its purpose (and the fact that
    opening it defeats the point of a hidden test set) obvious at a glance, and this
    whole tree is expected to be listed in .gitignore -- never committed. Scoring
    against it is a deliberate, one-time, end-of-project action (point evaluate's
    --gt-dir at ground_truth_DO_NOT_TOUCH/{id,ood} explicitly), not something that
    happens as a side effect of any routine pipeline command.
    """
    final_holdout_dir.mkdir(parents=True, exist_ok=True)
    for tag, df in (("id", final_holdout_id_df), ("ood", final_holdout_ood_df)):
        images_dir = final_holdout_dir / "images" / tag
        gt_dir = final_holdout_dir / "ground_truth_DO_NOT_TOUCH" / tag
        images_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df.iterrows():
            case_id = row["case_id"]
            shutil.copy2(row["t1w_path"], images_dir / f"{case_id}_0000.nii.gz")
            shutil.copy2(row["mask_path"], gt_dir / f"{case_id}.nii.gz")

        csv_path = final_holdout_dir / f"final_holdout_{tag}.csv"
        df[["case_id", "center", "lesion_volume_mm3", "size_bin"]].to_csv(csv_path, index=False)
        print(f"  final_holdout_{tag} n={len(df):4d}  sites={df['center'].nunique():2d}  -> {final_holdout_dir}")

    gitignore_path = final_holdout_dir / ".gitignore"
    gitignore_path.write_text("*\n", encoding="utf-8")
    print(f"\nWrote hidden final holdout under {final_holdout_dir} (self-.gitignore'd, never commit this tree)")


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
        "--final-holdout-ood-frac",
        type=float,
        default=FINAL_HOLDOUT_OOD_TARGET_FRAC,
        help="Target fraction of the whole usable pool for the hidden final_holdout_ood sites.",
    )
    parser.add_argument(
        "--final-holdout-id-frac",
        type=float,
        default=FINAL_HOLDOUT_ID_FRAC,
        help="Fraction of the non-final_holdout_ood pool to carve into final_holdout_id.",
    )
    parser.add_argument(
        "--final-holdout-ood-sites",
        default=None,
        help=(
            "Comma-separated site IDs to use as the hidden final_holdout_ood set, overriding the "
            f"guaranteed+search selection (guaranteed: {','.join(FINAL_HOLDOUT_OOD_GUARANTEED_SITES)})."
        ),
    )
    parser.add_argument(
        "--skip-final-holdout",
        action="store_true",
        help="Skip the hidden final_holdout_id/final_holdout_ood carve-out entirely (legacy behavior).",
    )
    parser.add_argument(
        "--final-holdout-dir",
        default=None,
        help=(
            "Where to write final_holdout_{id,ood}.csv plus image/ground-truth copies. Defaults to "
            "'<out-dir's parent>/final_holdout'. Deliberately separate from --out-dir and git-ignored "
            "-- see module docstring."
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

    # --- Stage 1: carve out the hidden final holdout FIRST, before anything else ever
    # sees these cases. This project's stand-in for the real (never-received) ISLES'26
    # challenge test set -- see module docstring.
    ood_sites: list[str] | None = None
    if args.skip_final_holdout:
        final_holdout_ood_df = usable.iloc[0:0].copy()
        final_holdout_id_df = usable.iloc[0:0].copy()
        pool_for_dev = usable
    else:
        if args.final_holdout_ood_sites:
            final_holdout_ood_sites = sorted(args.final_holdout_ood_sites.split(","))
        else:
            guaranteed = [s for s in FINAL_HOLDOUT_OOD_GUARANTEED_SITES if s in set(usable["center"])]
            missing = set(FINAL_HOLDOUT_OOD_GUARANTEED_SITES) - set(guaranteed)
            if missing:
                print(f"[warning] guaranteed final_holdout_ood sites not present in usable data: {sorted(missing)}")
            final_holdout_ood_sites = choose_site_pool(
                usable,
                target_frac=args.final_holdout_ood_frac,
                seed=args.seed,
                guaranteed_sites=guaranteed,
                excluded_sites=LOCKED_OOD_SITES,
            )
        print(f"\n[final holdout] OOD sites ({len(final_holdout_ood_sites)}): {final_holdout_ood_sites}")

        final_holdout_ood_df = usable[usable["center"].isin(final_holdout_ood_sites)].copy()
        pool_after_ood = usable[~usable["center"].isin(final_holdout_ood_sites)].copy()

        # Dev-time OOD sites must be known BEFORE the final_holdout_id carve below, so
        # that carve can exclude them -- see the excluded_sites docstring on
        # carve_final_holdout_id for why (an "in-distribution" hidden case must never
        # come from a site reserved for test_ood). This moves what used to be "Stage 2"
        # up a step; stratified_split (further down) still does the actual dev-time
        # split, unchanged.
        if args.ood_sites == "search":
            ood_sites = choose_ood_sites(pool_after_ood, target_frac=args.test_ood_frac, seed=args.seed)
            print(f"\n[search] selected {len(ood_sites)} OOD (held-out) sites: {ood_sites}")
        else:
            ood_sites = sorted(args.ood_sites.split(",")) if args.ood_sites else sorted(LOCKED_OOD_SITES)
            unknown = set(ood_sites) - set(pool_after_ood["center"].unique())
            if unknown:
                print(f"[warning] requested OOD sites not present in dev-time pool: {sorted(unknown)}")
            print(f"\nUsing fixed OOD (held-out) sites: {ood_sites}")

        final_holdout_id_df, pool_for_dev = carve_final_holdout_id(
            pool_after_ood, args.final_holdout_id_frac, args.seed, excluded_sites=ood_sites
        )
        print(
            f"[final holdout] ID sample: n={len(final_holdout_id_df)} "
            f"(from {len(pool_after_ood)} non-final_holdout_ood cases, excluding OOD-reserved sites)"
        )
        print(
            f"[final holdout] total hidden: {len(final_holdout_ood_df) + len(final_holdout_id_df)} "
            f"/ {len(usable)} usable cases -- these are excluded from every dev-time split below"
        )

    # --- Stage 2: existing dev-time split, applied only to what's left. ood_sites was
    # already determined above (needed earlier for the final_holdout_id exclusion);
    # the skip_final_holdout branch above never sets it, so it's determined here in
    # that case only.
    if ood_sites is None:
        if args.ood_sites == "search":
            ood_sites = choose_ood_sites(pool_for_dev, target_frac=args.test_ood_frac, seed=args.seed)
            print(f"\n[search] selected {len(ood_sites)} OOD (held-out) sites: {ood_sites}")
        else:
            ood_sites = sorted(args.ood_sites.split(",")) if args.ood_sites else sorted(LOCKED_OOD_SITES)
            unknown = set(ood_sites) - set(pool_for_dev["center"].unique())
            if unknown:
                print(f"[warning] requested OOD sites not present in dev-time pool: {sorted(unknown)}")
            print(f"\nUsing fixed OOD (held-out) sites: {ood_sites}")

    split_df = stratified_split(
        pool_for_dev,
        ood_sites=ood_sites,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_id_frac=args.test_id_frac,
        seed=args.seed,
    )

    # Leakage assertion: every dev-time case in exactly one split, and no overlap with
    # the hidden final holdout (carved out of a disjoint pool by construction, but
    # asserted explicitly since this is exactly the property that must never break).
    assert split_df["case_id"].is_unique
    assert set(split_df["split"].unique()) <= {"train", "val", "test_id", "test_ood"}
    hidden_ids = set(final_holdout_ood_df["case_id"]) | set(final_holdout_id_df["case_id"])
    assert not (set(split_df["case_id"]) & hidden_ids), "final holdout leaked into a dev-time split"

    if args.sample_per_split:
        split_df = subsample_per_split(split_df, args.sample_per_split, args.seed)
        print(f"\n[sample] shrunk to ~{args.sample_per_split} cases per split -> {len(split_df)} total")

    for split_name in ("train", "val", "test_id", "test_ood"):
        subset = split_df[split_df["split"] == split_name]
        path = out_dir / f"{split_name}.csv"
        subset[["case_id", "center", "lesion_volume_mm3", "size_bin"]].to_csv(path, index=False)
        print(f"  {split_name:9s} n={len(subset):4d}  sites={subset['center'].nunique():2d}  -> {path}")

    # manifest.csv deliberately covers only the four dev-time splits -- listing
    # final_holdout_* case IDs here (even without images/masks) would defeat the point
    # of "hidden": anyone opening this file would immediately see which cases they are.
    manifest_path = out_dir / "manifest.csv"
    split_df[["case_id", "center", "lesion_volume_mm3", "size_bin", "split"]].to_csv(manifest_path, index=False)
    print(f"\nWrote combined manifest (dev-time splits only): {manifest_path}")

    if not args.skip_final_holdout:
        final_holdout_dir = (
            Path(args.final_holdout_dir).expanduser().resolve()
            if args.final_holdout_dir
            else out_dir.parent / "final_holdout"
        )
        write_final_holdout(final_holdout_id_df, final_holdout_ood_df, final_holdout_dir)

    print("\nPer-split size-bin composition (fractions):")
    comp = pd.crosstab(split_df["split"], split_df["size_bin"], normalize="index")
    print(comp.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
