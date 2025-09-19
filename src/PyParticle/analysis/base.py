from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence, Tuple



@dataclass(frozen=True)
class VariableMeta:
    name: str
    value_key: str
    axis_keys: Sequence[str]
    description: str
    default_cfg: Dict[str, Any]
    aliases: Tuple[str, ...] = ()
    units: Dict[str, str] | None = None


from .population.base import PopulationVariable, VariableMeta

# Backwards compatible alias. Many existing factory modules import
# `AbstractVariable` from `analysis.base`; keep that name working.
AbstractVariable = PopulationVariable

__all__ = ["PopulationVariable", "AbstractVariable", "VariableMeta"]
