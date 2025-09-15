"""Builder helpers to create optical particles and optical populations.

This module wraps a morphology discovery registry to construct per-particle
optical objects and aggregate them into an `OpticalPopulation`.
"""

from .factory.registry import discover_morphology_types
from .base import OpticalPopulation


class OpticalParticleBuilder:
    """Construct an optical particle instance from a config.

    Parameters
    ----------
    config : dict
        Configuration dictionary with a 'type' key indicating morphology.
    """
    def __init__(self, config):
        self.config = config
    
    def build(self, base_particle):
        type_name = self.config.get("type")
        if not type_name:
            raise ValueError("Config must include a 'type' key.")
        types = discover_morphology_types()
        if type_name not in types:
            raise ValueError(f"Unknown optics morphology type: {type_name}")
        cls_or_factory = types[type_name]
        # Expect a class or callable that accepts (base_particle, config)
        return cls_or_factory(base_particle, self.config)


def build_optical_particle(base_particle, config):
    """Helper: build and return an optical particle from base particle and config."""
    return OpticalParticleBuilder(config).build(base_particle)


def build_optical_population(base_population, config):
    """Build an OpticalPopulation from a base ParticlePopulation and config.

    Parameters
    ----------
    base_population : ParticlePopulation
        Base population containing species, masses, concentrations, and ids.
    config : dict
        Optics configuration (rh_grid, wvl_grid, and morphology type).

    Returns
    -------
    OpticalPopulation
        Aggregated optics for the population.
    """
    rh_grid = config.get('rh_grid', [0.0])
    wvl_grid = config.get('wvl_grid', [550e-9])
    
    # Pass the base population so OpticalPopulation can inherit ids/num_concs/etc.
    optical_population = OpticalPopulation(base_population, rh_grid, wvl_grid)

    for part_id in base_population.ids:
        base_particle = base_population.get_particle(part_id)
        optical_particle = build_optical_particle(base_particle, config)
        optical_population.add_optical_particle(optical_particle, part_id)
        
    return optical_population