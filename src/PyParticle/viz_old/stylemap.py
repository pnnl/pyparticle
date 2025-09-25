"""Utilities to map scenario and series labels to colors and linestyles.

These functions are deterministic (optionally seeded) and return mappings used
by the plotting adapter.
"""
from __future__ import annotations

from typing import Sequence, Dict
import hashlib


def _stable_hash(s: str) -> int:
    return int(hashlib.sha1(s.encode("utf8")).hexdigest()[:8], 16)


def map_colors_for_scenarios(scenarios: Sequence[str], cmap: str = "tab10", seed: int | None = None) -> Dict[str, str]:
    """Return a deterministic mapping scenario->hex color string.

    Samples the matplotlib colormap evenly across the number of scenarios.
    ``seed`` is accepted for API parity but mapping is deterministic from
    scenario order; callers should sort scenarios for reproducible results.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except Exception:
        # Fallback simple palette
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        return {s: palette[i % len(palette)] for i, s in enumerate(scenarios)}

    n = max(1, len(scenarios))
    cmap_obj = plt.get_cmap(cmap)
    colors = [mpl.colors.to_hex(cmap_obj(i / max(1, n - 1))) for i in range(n)]
    return {s: colors[i % len(colors)] for i, s in enumerate(scenarios)}


def map_linestyles_for_series(series_labels: Sequence[str], linestyles: Sequence[str] | None = None) -> Dict[str, str]:
    """Return a deterministic mapping series_label->matplotlib linestyle string."""
    styles = list(linestyles) if linestyles else ["-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    mapping = {}
    for i, s in enumerate(series_labels):
        mapping[s] = styles[i % len(styles)]
    return mapping
