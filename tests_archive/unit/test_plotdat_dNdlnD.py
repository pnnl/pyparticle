import numpy as np
from PyParticle.analysis import build_variable

def _assert_plotdat(pd: dict):
    for key in ("x", "y", "labs", "xscale", "yscale"):
        assert key in pd, f"Missing PlotDat key: {key}"
    assert isinstance(pd["labs"], (list, tuple)) and len(pd["labs"]) >= 2
    assert pd["xscale"] in {"linear", "log"}
    assert pd["yscale"] in {"linear", "log"}
    x = np.asarray(pd["x"]); y = np.asarray(pd["y"])
    assert x.ndim == 1 and y.ndim == 1 and x.size == y.size

def test_plotdat_dNdlnD(population):
    var = build_variable("dNdlnD", scope="population", var_cfg={})
    vardat = var.compute(population, as_dict=True)
    pd = {"x": vardat.get("D"), "y": vardat.get("dNdlnD"),
          "labs": ["Diameter (m)", "dN/dlnD"],
          "xscale": "log", "yscale": getattr(var.meta, "scale", "linear")}
    _assert_plotdat(pd)
    x = np.asarray(pd["x"]); y = np.asarray(pd["y"])
    assert np.all(x > 0)
    assert np.all(np.isfinite(y))
    assert np.all(y >= 0)
