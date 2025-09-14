from .factory.registry import discover_morphology_types
from .base import OpticalPopulation

class OpticalParticleBuilder:
    def __init__(self, config):
        self.config = config
    
    def build(self, base_particle):
        # New API: callers should specify the morphology via the 'morphology' key
        morph_name = self.config.get("type")
        if not morph_name:
            raise ValueError("Config must include a 'morphology' key.")
        types = discover_morphology_types()
        if morph_name not in types:
            raise ValueError(f"Unknown optics morphology type: {morph_name}")
        cls_or_factory = types[morph_name]
        # Expect a class or callable that accepts (base_particle, config)
        return cls_or_factory(base_particle, self.config)

def build_optical_particle(base_particle, config):
    return OpticalParticleBuilder(config).build(base_particle)

def build_optical_population(base_population, config):
    """
    Build an OpticalPopulation from a base population.

    Strict API: requires a single `config` dict. Example:
        optics_cfg = {"rh_grid": [...], "wvl_grid": [...], "morphology": "core-shell", ...}
        build_optical_population(population, optics_cfg)
    """
    if not isinstance(config, dict):
        raise TypeError("build_optical_population requires a config dict as the second argument")

    rh_grid = config.get('rh_grid', [0.0])
    wvl_grid = config.get('wvl_grid', [550e-9])

    # Pass the base population so OpticalPopulation can inherit ids/num_concs/etc.
    optical_population = OpticalPopulation(base_population, rh_grid, wvl_grid)

    for part_id in base_population.ids:
        base_particle = base_population.get_particle(part_id)
        optical_particle = build_optical_particle(base_particle, config)
        optical_population.add_optical_particle(optical_particle, part_id)

    return optical_population