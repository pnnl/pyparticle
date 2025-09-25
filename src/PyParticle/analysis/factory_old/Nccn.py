from __future__ import annotations
import numpy as np
from ..base import AbstractVariable, VariableMeta
from .registry import register_variable

@register_variable("Nccn")
class NccnVar(AbstractVariable):
    meta = VariableMeta(
        name="Nccn",
        value_key="Nccn",
        axis_keys=("s",),
        description="CCN activation spectrum",
        # axis/grid defaults centralized in analysis.defaults. Keep only
        # variable-specific defaults here.
        default_cfg={"T": 298.15},
    )

    def compute(self, population):
        cfg = self.cfg
        s_eval = cfg["s_eval"]
        s_eval = np.asarray(s_eval, dtype=float)
        out = np.zeros_like(s_eval, dtype=float)
        for idx, s_env in enumerate(s_eval):
            c = 0.0
            for i, pid in enumerate(population.ids):
                part = population.get_particle(pid)
                s_crit = part.get_critical_supersaturation(cfg["T"], return_D_crit=False)
                if s_env >= s_crit:
                    c += float(population.num_concs[i])
            out[idx] = c
        return {"s": s_eval, "Nccn": out}


def build(cfg=None):
    cfg = cfg or {}
    return NccnVar(cfg)
