import types
from PyParticle.viz import data_prep


def test_prepare_dNdlnD_delegates(monkeypatch, small_binned_pop):
    called = {}

    def fake_compute(pop, name, cfg=None):
        called['args'] = (pop, name, cfg)
        return {"D": [1.0], "dNdlnD": [10.0]}

    # data_prep imports compute_variable as _compute_variable
    monkeypatch.setattr(data_prep, "_compute_variable", fake_compute)
    out = data_prep.prepare_dNdlnD(small_binned_pop, {"N_bins": 5})
    assert isinstance(out, dict)
    assert called['args'][1] == "dNdlnD"
