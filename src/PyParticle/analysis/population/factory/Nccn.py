from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

@register_variable("Nccn")
class NccnVar(PopulationVariable):
    meta = VariableMeta(
        name="Nccn",
            axis_names=("s",),
        description="CCN number concentration as a function of supersaturation",
        units="m$^{-3}$",
        aliases=(),
        scale='linear',
        long_label = 'CCN number concentration',
        short_label = '$N_{\mathrm{CCN}}(s)$',
        # s-grid default centralized in analysis.defaults; keep other defaults
        default_cfg={"T": 298.15},
    )
    
    def compute(self, population, as_dict=False):
        cfg = self.cfg
        s_eval = cfg.get("s_eval", [])
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
        if as_dict:
            return {"s": s_eval, "Nccn": out}
        return out


# def build(cfg=None):
#     cfg = cfg or {}
#     return NccnVar(cfg)
