"""Lean analysis prototype (Option A).

Contrasts with the layered architecture in `PyParticle.analysis` by collapsing
numerical logic & variable metadata into single factory modules.

Public functions mirror layered version for comparison:
  - compute_variable
  - list_variables

Intended for evaluation only; not wired into visualization layer by default.
"""
#from .dispatcher import compute_variable, list_variables, describe_variable
from .builder import build_variable
# Eagerly import factory modules so decorators and build functions are registered.
from importlib import import_module
import pkgutil, os
_pkg_path = os.path.join(os.path.dirname(__file__), "population", "factory")
for _, module_name, _ in pkgutil.iter_modules([_pkg_path]):
  if module_name in ("__init__", "registry"):
    continue
  import_module(f"{__package__}.population.factory.{module_name}")

# # --- Plot data preparers -------------------------------------------------------
# from .prepare_old import (
#     compute_plotdat,
#     build_default_var_cfg,
#     register_preparer,
# )

__all__ = [
  "build_variable",
  "compute_variable"
  "lists_variables"
  "describe_variable"]
#     "compute_plotdat",
#     "build_default_var_cfg",
#     "register_preparer",
# ]