from .core_shell import CoreShellParticle
from .homogeneous import HomogeneousParticle
from .fractal import FractalAggregateParticle

MORPHOLOGY_REGISTRY = {
    "core-shell": CoreShellParticle,
    "homogeneous": HomogeneousParticle,
    "fractal": FractalAggregateParticle,
}

def create_optical_particle(morphology, *args, **kwargs):
    """
    Factory function for instantiating an optical particle model by morphology.
    """
    if morphology not in MORPHOLOGY_REGISTRY:
        raise ValueError(f"Unknown morphology: {morphology}")
    return MORPHOLOGY_REGISTRY[morphology](*args, **kwargs)
