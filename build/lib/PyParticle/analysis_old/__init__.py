"""Backup of the original analysis package (pre-refactor).

This module is created as a safe fallback. It re-exports the original
public functions from `analysis` when needed. It is not used by default.
"""
from importlib import import_module
import pkgutil, os

# Re-export top-level analysis dispatchers from the current analysis package
try:
    from PyParticle.analysis.dispatcher import compute_variable, list_variables, describe_variable
except Exception:
    # best-effort: leave names undefined if import fails
    compute_variable = None
    list_variables = None
    describe_variable = None

__all__ = ["compute_variable", "list_variables", "describe_variable"]
