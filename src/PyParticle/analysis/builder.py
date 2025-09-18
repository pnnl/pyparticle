from __future__ import annotations
from typing import Dict, Any, Callable
from .factory.registry import get_builder


class VariableBuilder:
    def __init__(self, name: str, cfg: Dict[str, Any] | None = None):
        self.name = name
        self.cfg = cfg or {}
        self._mods: Dict[str, Any] = {}

    def modify(self, **k):
        self._mods.update(k)
        return self

    def build(self):
        builder: Callable = get_builder(self.name)
        # Attempt to get default config from builder.meta or from an instance
        defaults: Dict[str, Any] = {}
        if hasattr(builder, "meta"):
            defaults = dict(getattr(builder, "meta").default_cfg)
        else:
            try:
                inst = builder({})
                defaults = dict(getattr(inst, "meta").default_cfg)
            except Exception:
                defaults = {}

        merged = dict(defaults)
        merged.update(self.cfg)
        merged.update(self._mods)

        # Call the builder with merged cfg
        obj = builder(merged)
        return obj


def build_variable(name: str, **cfg):
    return VariableBuilder(name, cfg).build()
