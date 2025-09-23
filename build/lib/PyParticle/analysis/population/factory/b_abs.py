from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable
from PyParticle.optics.builder import build_optical_population

@register_variable("b_abs")
class BAbsVar(PopulationVariable):
    meta = VariableMeta(
        name="b_abs",
        value_key="b_abs",
        axis_keys=("rh_grid", "wvls"),
        description="Absorption coefficient",
        default_cfg={
            "wvls": [550e-9],
            "rh_grid": [0.0, 0.5, 0.9],
            "morphology": "core-shell",
            "species_modifications": {},
            "T": 298.15,
        },
        aliases=("total_abs",),
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
        arr = optical_pop.get_optical_coeff("b_abs", rh=None, wvl=None)
        return {"rh_grid": np.asarray(cfg["rh_grid"]), "wvls": np.asarray(cfg["wvls"]), "b_abs": arr}


def build(cfg=None):
    cfg = cfg or {}
    return BAbsVar(cfg)
