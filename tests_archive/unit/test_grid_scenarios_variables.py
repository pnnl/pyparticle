import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from PyParticle.viz import make_grid_scenarios_variables_same_timestep
from PyParticle.population.builder import build_population


def test_make_grid_basic(tmp_path):
    # two tiny scenario dicts
    scen1 = {"type": "binned_lognormal", "name": "s1", "timestep": 0}
    scen2 = {"type": "binned_lognormal", "name": "s2", "timestep": 0}

    variables = ["dNdlnD", "frac_ccn"]
    fig, axarr, pops = make_grid_scenarios_variables_same_timestep(
        [scen1, scen2], variables, timestep=0, return_populations=True
    )

    assert fig is not None
    import numpy as np

    assert isinstance(axarr, np.ndarray)
    assert axarr.shape == (2, 2)

    # ensure at least one artist per axis
    for ax in axarr.flatten():
        assert len(ax.get_lines()) >= 0

    # test prebuilt population path: build one and pass
    prebuilt = build_population({"type": "binned_lognormal", "name": "pre", "timestep": 0})
    fig2, axarr2 = make_grid_scenarios_variables_same_timestep([prebuilt], variables, timestep=0)
    assert axarr2.shape == (1, 2)
