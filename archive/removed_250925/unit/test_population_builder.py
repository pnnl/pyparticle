from __future__ import annotations
import numpy as np


def test_build_binned_lognormal_population(population, small_cfg):
    n_bins = int(small_cfg["population"]["N_bins"])
    # The binned_lognormals builder represents each bin as a particle id.
    assert hasattr(population, "ids")
    assert len(population.ids) == n_bins
    # spec_masses should have shape (n_bins, n_species)
    assert hasattr(population, "spec_masses")
    sm = np.asarray(population.spec_masses)
    assert sm.ndim == 2 and sm.shape[0] == n_bins

    # Reconstruct a diameter array by querying each particle's dry diameter
    Ds = []
    for pid in population.ids:
        p = population.get_particle(pid)
        Ds.append(p.get_Ddry())
    Ds = np.asarray(Ds)
    # diameters should be strictly positive and increasing
    assert np.all(Ds > 0)
    assert np.all(np.diff(Ds) >= 0), "Diameter sequence should be non-decreasing"
