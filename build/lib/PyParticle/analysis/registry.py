"""Compatibility shim: re-export registry functions from analysis.population.factory.registry.

The authoritative registry now lives in `analysis.population.factory.registry`.
This shim preserves `from PyParticle.analysis.registry import ...` imports while
keeping the canonical location under `analysis/population/factory`.
"""
from .population.factory.registry import *  # noqa: F401,F403
