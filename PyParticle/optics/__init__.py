from .factory import create_optical_particle, MORPHOLOGY_REGISTRY
from .base import OpticalParticle
from .utils import get_cross_section_array_from_population, OPTICS_TYPE_MAP
from .population import OpticalPopulation
from .refractive_index import RefractiveIndex, RI_fun

def make_optical_particle(
    particle,
    rh_grid,
    wvl_grid,
    morphology="core-shell",
    compute_optics=True,
    temp=293.15,
    specdata_path=None,
    species_modifications=None,
    **kwargs
):
    """
    Factory to create and optionally compute optics for an optical particle.
    """
    optical_particle = create_optical_particle(
        morphology,
        particle.species,
        particle.masses,
        rh_grid=rh_grid,
        wvl_grid=wvl_grid,
        temp=temp,
        specdata_path=specdata_path,
        species_modifications=species_modifications,
        **kwargs
    )
    if compute_optics:
        optical_particle.compute_optics()
    return optical_particle

def make_optical_population(
    particle_population,
    rh_grid,
    wvl_grid,
    morphology="core-shell",
    compute_optics=True,
    temp=293.15,
    specdata_path=None,
    species_modifications=None,
    **kwargs
):
    """
    Factory to create an OpticalPopulation from a particle population object
    (which is expected to have .particles and .num_concs attributes).
    """
    optical_population = OpticalPopulation(rh_grid, wvl_grid)
    # Accept both a "particles" attribute or an iterable passed directly
    particles = getattr(particle_population, "particles", particle_population)
    num_concs = getattr(particle_population, "num_concs", None)
    if num_concs is None:
        # Try to get from attribute or parallel list/array argument
        num_concs = kwargs.get("num_concs", [1.0] * len(particles))
    for particle, num_conc in zip(particles, num_concs):
        optical_particle = make_optical_particle(
            particle,
            rh_grid,
            wvl_grid,
            morphology=morphology,
            compute_optics=compute_optics,
            temp=temp,
            specdata_path=specdata_path,
            species_modifications=species_modifications,
            **kwargs
        )
        optical_population.particles.append(optical_particle)
        optical_population.num_concs.append(num_conc)
        optical_population.ids.append(getattr(particle, "id", len(optical_population.ids)))
    return optical_population
