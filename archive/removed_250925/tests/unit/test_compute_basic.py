from PyParticle.analysis import compute_variable


def test_compute_dNdlnD_small_population(small_binned_pop):
    pop = small_binned_pop
    res = compute_variable(pop, "dNdlnD", {"N_bins": 10, "D_min": 1e-9, "D_max": 2e-6})
    assert "D" in res and "dNdlnD" in res
    assert len(res["D"]) == 10
    assert len(res["dNdlnD"]) == 10
