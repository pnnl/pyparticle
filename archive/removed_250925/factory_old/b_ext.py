from __future__ import annotations
import numpy as np
from ..base import AbstractVariable, VariableMeta
from .registry import register_variable
from ...optics.builder import build_optical_population

@register_variable("b_ext")
class BExtVar(AbstractVariable):
    meta = VariableMeta(
        name="b_ext",
        value_key="b_ext",
        axis_keys=("rh_grid", "wvls"),
        description="Extinction coefficient",
        # axis/grid defaults centralized in analysis.defaults
        default_cfg={
            "morphology": "core-shell",
            "species_modifications": {},
            "T": 298.15,
        },
        aliases=("total_ext",),
    )

    def compute(self, population):
        cfg = self.cfg
        morph = cfg["morphology"]
        if morph == "core-shell":
            morph = "core_shell"
        ocfg = {
            "rh_grid": list(cfg["rh_grid"]),
            "wvl_grid": list(cfg["wvls"]),
            "type": morph,
            "temp": cfg["T"],
            "species_modifications": cfg.get("species_modifications", {}),
        }
        optical_pop = build_optical_population(population, ocfg)
        arr = optical_pop.get_optical_coeff("b_ext", rh=None, wvl=None)
        return {"rh_grid": np.asarray(cfg["rh_grid"]), "wvls": np.asarray(cfg["wvls"]), "b_ext": arr}


def build(cfg=None):
    cfg = cfg or {}
    return BExtVar(cfg)
