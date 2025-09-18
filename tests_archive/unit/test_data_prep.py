import numpy as np
import types

from PyParticle.viz import data_prep


class FakeParticle:
    def __init__(self, Dwet, Ddry, kappa, s_crit):
        self._Dwet = Dwet
        self._Ddry = Ddry
        self._kappa = kappa
        self._s_crit = s_crit

    def get_Dwet(self):
        return self._Dwet

    def get_Ddry(self):
        return self._Ddry

    def get_tkappa(self):
        return self._kappa

    def get_critical_supersaturation(self, T, return_D_crit=False):
        return self._s_crit


class FakePopulation:
    def __init__(self, particles, num_concs):
        self.ids = list(range(len(particles)))
        self._particles = {i: p for i, p in enumerate(particles)}
        self.num_concs = np.array(num_concs, dtype=float)

    def get_particle(self, pid):
        return self._particles[pid]


def test_prepare_dNdlnD_basic():
    parts = [FakeParticle(50e-9, 45e-9, 0.3, 0.01), FakeParticle(150e-9, 140e-9, 0.2, 0.02)]
    pop = FakePopulation(parts, [1.0, 2.0])
    cfg = {"wetsize": True, "N_bins": 10, "D_min": 10e-9, "D_max": 1e-6}
    out = data_prep.prepare_dNdlnD(pop, cfg)
    assert "x" in out and "y" in out
    assert out["x"].shape[0] == out["y"].shape[0]
    assert out["xscale"] == "log"


def test_prepare_Nccn_and_frac():
    parts = [FakeParticle(50e-9, 45e-9, 0.3, 0.001), FakeParticle(150e-9, 140e-9, 0.2, 0.02)]
    pop = FakePopulation(parts, [1.0, 1.0])
    cfg = {"s_eval": np.array([0.0005, 0.001, 0.01])}
    out = data_prep.prepare_Nccn(pop, cfg)
    assert out["x"].shape[0] == 3
    outf = data_prep.prepare_frac_ccn(pop, cfg)
    assert np.all(outf["y"] <= 1.0)


def test_prepare_optical_vs_wvl_monkeypatch(monkeypatch):
    # monkeypatch optics builder used inside data_prep.compute_optical_coeffs
    def fake_build_optical_population(population, cfg):
        class FakeOpt:
            def get_optical_coeff(self, optics_type, rh, wvl):
                # return a simple 2x3 array
                return np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        return FakeOpt()

    monkeypatch.setattr('PyParticle.viz.data_prep.build_optical_population', fake_build_optical_population, raising=False)

    parts = [FakeParticle(50e-9, 45e-9, 0.3, 0.001)]
    pop = FakePopulation(parts, [1.0])
    cfg = {"wvls": np.array([400e-9, 550e-9, 700e-9]), "rh_grid": np.array([0.0, 0.5])}
    out = data_prep.prepare_optical_vs_wvl(pop, {"coeff": "total_ext", "wvls": cfg["wvls"], "rh_grid": cfg["rh_grid"]})
    assert out["x"].shape[0] == 3
    assert out["y"].shape[0] == 3
