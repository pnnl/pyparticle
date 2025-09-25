# viz/style.py
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple, Iterable, Mapping, Any
import hashlib

# Shared defaults
DEFAULT_PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]
DEFAULT_LINESTYLES = ["-","--","-.",":"]
DEFAULT_MARKERS = ["o","s","^","D","v","P","X"]

@dataclass
class GeomDefaults:
    # Discrete cycles
    palette: List[str] = field(default_factory=lambda: DEFAULT_PALETTE.copy())
    linestyles: List[str] = field(default_factory=lambda: DEFAULT_LINESTYLES.copy())
    markers: List[str] = field(default_factory=lambda: DEFAULT_MARKERS.copy())
    # Scalar defaults
    linewidth: float = 2.0
    markersize: float = 36.0  # matplotlib 's' is area in points^2
    alpha: float | None = None
    # Continuous mappings
    cmap: str = "viridis"  # for scatter/surface color mapping
    
    # how to combine when both color and something else cycle
    def combos(self, use_linestyle: bool, use_marker: bool) -> List[Tuple[str, str | None, str | None]]:
        # (color, linestyle, marker)
        if use_linestyle and use_marker:
            return [(c, ls, mk) for c, ls, mk in product(self.palette, self.linestyles, self.markers)]
        if use_linestyle:
            return [(c, ls, None) for c, ls in product(self.palette, self.linestyles)]
        if use_marker:
            return [(c, None, mk) for c, mk in product(self.palette, self.markers)]
        return [(c, None, None) for c in self.palette]

@dataclass
class Theme:
    # Per-geometry defaults; extend as you add geoms
    geoms: Dict[str, GeomDefaults] = field(default_factory=lambda: {
        "line": GeomDefaults(linewidth=2.0, alpha=None),
        "scatter": GeomDefaults(linewidth=1.0, markersize=36.0),
        "bar": GeomDefaults(),
        "box": GeomDefaults(),
        "surface": GeomDefaults(),
    })

class StyleManager:
    """
    Plans per-series matplotlib kwargs given a geometry and series keys.
    Deterministic mapping: same key → same style across figures.
    """
    def __init__(self, theme: Theme | None = None, deterministic: bool = True):
        self.theme = theme or Theme()
        self.deterministic = deterministic

    def _index_for_key(self, key: str, i: int) -> int:
        if not self.deterministic:
            return i
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def plan(
        self,
        geom: str,
        series_keys: Iterable[str],
        *,
        overrides: Mapping[str, Dict[str, Any]] | None = None,
        cycle_linestyle: bool | None = None,
        cycle_marker: bool | None = None,
    ) -> Dict[str, Dict[str, Any]]:
        if geom not in self.theme.geoms:
            raise ValueError(f"Unknown geom '{geom}'. Known: {list(self.theme.geoms)}")
        gd = self.theme.geoms[geom]
        # Sensible defaults per geom
        use_ls = gd.linestyles and (cycle_linestyle if cycle_linestyle is not None else geom == "line")
        use_mk = gd.markers and (cycle_marker if cycle_marker is not None else geom == "scatter")
        combos = gd.combos(use_ls, use_mk)
        ncombo = len(combos)

        styles: Dict[str, Dict[str, Any]] = {}
        for i, key in enumerate(series_keys):
            idx = self._index_for_key(key, i) % ncombo
            color, linestyle, marker = combos[idx]
            base: Dict[str, Any] = {"color": color}
            if use_ls and linestyle is not None:
                base["linestyle"] = linestyle
                base["linewidth"] = gd.linewidth
            if use_mk and marker is not None:
                base["marker"] = marker
                base["s"] = gd.markersize
            if gd.alpha is not None:
                base["alpha"] = gd.alpha
            # Continuous mappings use cmap by name; plotters will apply if needed
            base["cmap"] = gd.cmap
            if overrides and key in overrides:
                base.update(overrides[key])
            styles[key] = base
        return styles
