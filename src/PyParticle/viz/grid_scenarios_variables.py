"""Grid helper: rows=scenarios, cols=variables at a single common timestep.

This module implements `make_grid_scenarios_variables_same_timestep` which
builds populations (from dicts, paths or prebuilt objects), lays out a
matplotlib grid and calls the repository plotting utilities per cell.

Do NOT duplicate plotting logic — this helper delegates to
`PyParticle.population.build_population`, `PyParticle.analysis.build_default_var_cfg`,
`PyParticle.viz.plot_lines`, and formatting helpers.
"""
from __future__ import annotations

from typing import Sequence, Union, Optional, Mapping, Any, Tuple, Callable, List
import json
import os
import hashlib
from pathlib import Path

import numpy as np

from .layout import make_grid
from .plotting import plot_lines
from .formatting import format_axes, add_legend
from ..population.builder import build_population
from ..population.base import ParticlePopulation
from ..analysis import build_default_var_cfg


def _load_config_from_path(p: str) -> dict:
    p = Path(p)
    if not p.exists():
        raise ValueError(f"Scenario file not found: {p}")
    txt = p.read_text()
    try:
        return json.loads(txt)
    except Exception:
        # try yaml if yaml is available
        try:
            import yaml

            return yaml.safe_load(txt)
        except Exception:
            raise ValueError(f"Could not parse scenario file: {p}")


def _hash_dict(d: dict) -> str:
    # stable-ish hash for small dicts used as cache key
    txt = json.dumps(d, sort_keys=True)
    return hashlib.sha1(txt.encode("utf8")).hexdigest()


def make_grid_scenarios_variables_same_timestep(
    scenarios: Sequence[Union[dict, str, object]],
    variables: Sequence[str],
    timestep: Union[int, float],
    *,
    var_cfg_overrides: Optional[Mapping[str, dict]] = None,
    scenario_labels: Optional[Union[Sequence[str], Callable[[Any, int], str]]] = None,
    variable_titles: Optional[Sequence[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hspace: Optional[float] = None,
    wspace: Optional[float] = None,
    layout_spec: Optional[Sequence[Sequence[int]]] = None,
    return_populations: bool = False,
):
    """Build a grid with rows=scenarios and cols=variables at a single timestep.

    See repo prompt for full contract. Returns (fig, axarr) or (fig, axarr, pops).
    """

    rows = len(scenarios)
    cols = len(variables)

    if figsize is None:
        figsize = (max(4 * cols, 4), max(3 * rows, 3))

    fig, axarr = make_grid(rows, cols, figsize=figsize, layout_spec=layout_spec,
                           hspace=(0.3 if hspace is None else hspace),
                           wspace=(0.3 if wspace is None else wspace),
                           sharex=True, sharey=False)

    # normalize axarr to NumPy array shape (rows, cols)
    axarr = np.asarray(axarr)

    # tiny population cache
    pop_cache = {}
    populations: List[Any] = []

    for i, scen in enumerate(scenarios):
        cache_key = None
        pop = None
        if isinstance(scen, ParticlePopulation):
            # prebuilt ParticlePopulation instance: use as-is (ignore timestep)
            pop = scen
            cache_key = (id(pop), timestep)
        elif isinstance(scen, dict):
            cfg = dict(scen)  # shallow copy
            cfg["timestep"] = timestep
            cache_key = (_hash_dict(cfg), timestep)
            if cache_key in pop_cache:
                pop = pop_cache[cache_key]
            else:
                pop = build_population(cfg)
                pop_cache[cache_key] = pop
        elif isinstance(scen, str):
            doc = _load_config_from_path(scen)
            if isinstance(doc, dict) and "population" in doc:
                cfg = dict(doc["population"])
            elif isinstance(doc, dict):
                cfg = dict(doc)
            else:
                raise ValueError(f"Scenario file did not contain a mapping: {scen}")
            cfg["timestep"] = timestep
            cache_key = (_hash_dict(cfg), timestep)
            if cache_key in pop_cache:
                pop = pop_cache[cache_key]
            else:
                pop = build_population(cfg)
                pop_cache[cache_key] = pop
        else:
            raise TypeError("Scenario entries must be dict, path string, or prebuilt population")

        populations.append(pop)

    # plotting loop
    for i, pop in enumerate(populations):
        for j, var in enumerate(variables):
            ax = axarr[i, j]
            # prepare var cfg
            vcfg = build_default_var_cfg(var)
            if var_cfg_overrides and var in var_cfg_overrides:
                # shallow merge
                for k, v in var_cfg_overrides[var].items():
                    vcfg[k] = v

            try:
                artist, labs = plot_lines(var, (pop,), vcfg, ax=ax)
            except Exception as e:
                # make error actionable for dataset/species problems
                if "Species data" in str(e) or "not found" in str(e):
                    raise ValueError(
                        "Scenario plotting failed; check PYPARTICLE_DATA_PATH or species names: "
                        + str(e)
                    )
                else:
                    raise

            xlabel, ylabel = (None, None)
            if labs is not None and isinstance(labs, (list, tuple)) and len(labs) >= 2:
                xlabel, ylabel = labs[0], labs[1]

            # derive title and row label
            title = None
            if i == 0:
                title = variable_titles[j] if variable_titles is not None else var

            # leftmost column: add row/scene label
            if j == 0:
                rowlabel = None
                if isinstance(scenario_labels, (list, tuple)):
                    rowlabel = scenario_labels[i]
                elif callable(scenario_labels):
                    rowlabel = scenario_labels(scenarios[i], i)
                else:
                    # try derive from str path or population metadata
                    s = scenarios[i]
                    if isinstance(s, str):
                        rowlabel = Path(s).stem
                    elif isinstance(s, dict):
                        rowlabel = s.get("name") or s.get("id") or f"scenario_{i}"
                    else:
                        rowlabel = getattr(s, "name", None) or f"scenario_{i}"
                # prepend rowlabel to title if present otherwise set ylabel
                if title:
                    title = f"{rowlabel} — {title}"
                else:
                    title = rowlabel

            # apply formatting and legend
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)
            try:
                add_legend(ax)
            except Exception:
                pass

    if return_populations:
        return fig, axarr, populations
    return fig, axarr
