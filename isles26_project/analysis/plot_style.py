#!/usr/bin/env python3
"""Shared color/label conventions for cross-trainer comparison figures.

Not a script -- imported by analysis/plot_results.py and
analysis/plot_learning_curves.py so the same trainer always gets the same
color and the same short display label in every figure in the report,
instead of each script picking colors independently (which previously meant
e.g. WideAugBaseline could be blue in one figure and orange in another).

Non-obvious rationale: colors are assigned by sorting the full set of
trainer/experiment names alphabetically first, then cycling a fixed
qualitative palette over that sorted order. This is deterministic given a
name set (not insertion order, not dict iteration order), so adding or
dropping one trainer only reshuffles colors for names that sort after it,
not the whole palette -- and, more importantly, matches
plot_learning_curves.py's own discover_runs_from_index, which already
iterates `sorted(fold_dirs.items())` -- reusing that same sort order here
keeps the two scripts' palettes identical for the trainer set they share.

Usage:
    from plot_style import color_for, short_label, build_palette
    palette = build_palette(df["experiment"].unique())
    colors = [color_for(name, palette) for name in df["experiment"]]
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt

# tab10 first (bold, most distinguishable) then tab20's remaining entries --
# covers up to 20 trainers before any repeat; this project has had at most 11.
_BASE_CYCLE = [plt.cm.tab10(i) for i in range(10)] + [plt.cm.tab20(i) for i in range(20) if i % 2 == 1]

# Prefixes/suffixes stripped when building a short display label -- purely
# cosmetic (axis/legend text), never applied to the underlying data. Covers
# both naming conventions actually seen in this project's CSVs:
# plot_learning_curves.py's runs_index.csv uses "trainer (plans)" (parens);
# evaluate/aggregate's own `experiment` column uses "trainer__plans" (double
# underscore) -- both map to the same "(ResEncM)" short form.
_STRIP_PATTERNS = [
    (re.compile(r"^nnUNetTrainer"), ""),
    # NOTE: no trailing \b on "_500epochs" -- \b only matches between a word
    # char and a non-word char, and "_" (as in "_500epochs__nnUNetResEnc...")
    # IS a word char, so a \b-anchored pattern silently failed to strip
    # "_500epochs" whenever it was immediately followed by the ResEncM
    # suffix (e.g. "Baseline_500epochs__nnUNetResEncUNetMPlans" kept
    # "_500epochs" in the short label). Matching the literal substring
    # regardless of what follows is exactly what's wanted here.
    (re.compile(r"_500epochs_full"), ""),
    (re.compile(r"_500epochs"), ""),
    (re.compile(r"_1000epochs\b"), " (1000ep)"),
    (re.compile(r"\s*\(nnUNetResEncUNetMPlans\)"), " (ResEncM)"),
    (re.compile(r"__?nnUNetResEncUNetMPlans\b"), " (ResEncM)"),
]


def short_label(name: str) -> str:
    """'nnUNetTrainerWideAugBaseline_500epochs (nnUNetResEncUNetMPlans)' ->
    'WideAugBaseline (ResEncM)' -- cuts the boilerplate every trainer name
    shares so axis/legend text carries only the part that differs.

    Ensemble combos ('ensemble_A+B+C+D', from ensembling/ensemble_test.py)
    are shortened member-by-member and re-joined -- otherwise a 4-member
    combo's raw name alone is ~90 characters, wide enough to squeeze a
    whole figure's plot area down to a sliver to make room for the legend.
    """
    if name.startswith("ensemble_"):
        members = name[len("ensemble_"):].split("+")
        return "Ens: " + "+".join(short_label(m) for m in members)
    label = name
    for pattern, replacement in _STRIP_PATTERNS:
        label = pattern.sub(replacement, label)
    return label.strip()


def build_palette(names) -> dict[str, tuple]:
    """{name: color} for the given names, sorted alphabetically before
    assignment (see module docstring for why sorted, not insertion order)."""
    ordered = sorted(set(names))
    return {name: _BASE_CYCLE[i % len(_BASE_CYCLE)] for i, name in enumerate(ordered)}


def color_for(name: str, palette: dict[str, tuple]) -> tuple:
    return palette[name]
