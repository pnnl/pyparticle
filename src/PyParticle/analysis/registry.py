"""Compatibility shim: re-export registry functions from analysis.factory.registry.

The authoritative registry now lives in `analysis.factory.registry` to match the
population/optics package layout. This shim preserves `from PyParticle.analysis.registry import ...`
imports while keeping the canonical location under `analysis/factory/`.
"""
from .factory.registry import *  # noqa: F401,F403
