import numpy as np

def test_build_binned_lognormal_population_shape(population, small_cfg):
    n_bins = int(small_cfg["population"]["N_bins"])
    assert hasattr(population, "ids")
    assert len(population.ids) == n_bins

    sm = np.asarray(population.spec_masses)
    assert sm.ndim == 2 and sm.shape[0] == n_bins

    Ds = np.asarray([population.get_particle(pid).get_Ddry() for pid in population.ids])
    assert np.all(Ds > 0)
    assert np.all(np.diff(Ds) >= 0), "Diameter sequence should be non-decreasing"
    assert np.isclose(population.get_Ntot(), np.sum(np.array(small_cfg["population"]["N"])))

# def test_numdist_integrates_to_Ntot(population):
#     edges, num_counts = population.get_num_dist_1d(varname="wet_diameter", N_bins=40, density=False, weights=population.num_concs)
#     print(num_counts.sum(), population.get_Ntot())
#     # Basic contract: histogram returns a counts array with the expected length and non-negative entries
#     assert num_counts.ndim == 1
#     assert num_counts.size == 40
#     assert (num_counts >= 0).all()
