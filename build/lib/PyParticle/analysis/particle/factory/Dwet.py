from __future__ import annotations
from ..base import ParticleVariable, VariableMeta
from .registry import register_particle_variable


@register_particle_variable("Dwet")
class DwetVar(ParticleVariable):
    meta = VariableMeta(
        name="Dwet",
        value_key="Dwet",
        axis_keys=("D",),
        description="Wet diameter of a particle",
        default_cfg={},
    )

    def compute(self, particle):
        D = particle.get_Dwet()
        return {"D": D, "Dwet": D}


def build(cfg=None):
    cfg = cfg or {}
    return DwetVar(cfg)
