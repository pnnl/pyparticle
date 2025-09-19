from __future__ import annotations
from typing import Dict, Any, Callable

# Unified builder: routes to population or particle registries

def _get_registry_builder(scope: str) -> Callable[[str], Callable]:
	"""Return a function that, given a variable name, returns the builder callable.

	scope: either 'population' or 'particle'
	The returned callable has signature get_builder(name: str) -> callable
	where the callable is a builder that when called with cfg returns the variable instance.
	"""
	if scope == "population":
		from .population.factory.registry import get_population_builder as _g
		return _g
	if scope == "particle":
		from .particle.factory.registry import get_particle_builder as _g
		return _g
	raise ValueError(f"Unknown scope '{scope}'")


class VariableBuilder:
	"""Unified VariableBuilder that can build population or particle variables.

	Usage:
	  VariableBuilder(name, cfg=None, scope='population').build()
	"""
	def __init__(self, name: str, cfg: Dict[str, Any] | None = None, scope: str = "population"):
		self.name = name
		self.cfg = cfg or {}
		self._mods: Dict[str, Any] = {}
		self.scope = scope

	def modify(self, **k):
		self._mods.update(k)
		return self

	def build(self):
		get_builder = _get_registry_builder(self.scope)
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


def build_variable(name: str, scope: str = "population", **cfg):
	return VariableBuilder(name, cfg, scope=scope).build()


__all__ = ["VariableBuilder", "build_variable"]
