from __future__ import annotations
import numpy as np
from ..base import PopulationVariable, VariableMeta
from .registry import register_variable

@register_variable("frac_ccn")
class FracCCNVar(PopulationVariable):
    # fixme: rethink how VariableMeta is constructed? Use config?
    meta = VariableMeta(
        name="frac_ccn",
        axis_names=("s",),
        description="fraction of particles that are CCN-activate at given supersaturation",
        units="",
        short_label="$frac_{\mathrm{CCN}}$",
        long_label="fraction CCN-active",
        scale='linear',
        # s-grid default centralized in analysis.defaults; keep other defaults

    )

    def compute(self, population, as_dict=False):
        cfg = self.cfg
        s_eval = np.asarray(cfg.get("s_eval", cfg.get("s_grid", [])), dtype=float)
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
        if as_dict:
            return {"s": s_eval, "frac_ccn": frac}
        return frac


def build(cfg=None):
    cfg = cfg or {}
    return FracCCNVar(cfg)
