from __future__ import annotations
from typing import Dict, Any

# Reuse VariableMeta from population base to avoid duplication
from ..population.base import VariableMeta


class ParticleVariable:
    """Base class for variables that operate on a single particle.

    Implementations should provide `meta: VariableMeta` and a `compute(particle)`
    method that returns a dict mapping axis keys and the value keyed by
    `meta.value_key`.
    """
    meta: VariableMeta

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def compute(self, particle):  # pragma: no cover - interface
        raise NotImplementedError

