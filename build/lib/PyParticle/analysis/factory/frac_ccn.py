from __future__ import annotations
import numpy as np
from ..base import AbstractVariable, VariableMeta
from .registry import register_variable

@register_variable("frac_ccn")
class FracCCNVar(AbstractVariable):
    meta = VariableMeta(
        name="frac_ccn",
        value_key="frac_ccn",
        axis_keys=("s",),
        description="Fractional CCN activation",
        default_cfg={"s_eval": np.linspace(0.01, 1.0, 50), "T": 298.15},
    )

    def compute(self, population):
        cfg = self.cfg
        s_eval = np.asarray(cfg["s_eval"], dtype=float)
        # reuse logic from NccnVar
        nccn = []
        for s_env in s_eval:
            c = 0.0
            for i, pid in enumerate(population.ids):
                part = population.get_particle(pid)
                s_crit = part.get_critical_supersaturation(cfg["T"], return_D_crit=False)
                if s_env >= s_crit:
                    c += float(population.num_concs[i])
            nccn.append(c)
        nccn = np.asarray(nccn)
        total = float(sum(population.num_concs))
        frac = nccn / total if total > 0 else nccn
        return {"s": s_eval, "frac_ccn": frac}


def build(cfg=None):
    cfg = cfg or {}
    return FracCCNVar(cfg)
