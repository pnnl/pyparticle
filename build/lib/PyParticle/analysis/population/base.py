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


class PopulationVariable:
    """Base class for variables that operate on an entire population.

    This class replaces the legacy `AbstractVariable`. For backwards
    compatibility the top-level `analysis.base` module will alias
    `AbstractVariable = PopulationVariable`.
    """
    meta: VariableMeta

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def compute(self, population):  # pragma: no cover - interface
        raise NotImplementedError

