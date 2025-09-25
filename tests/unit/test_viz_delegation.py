from PyParticle.analysis import compute_plotdat


def test_prepare_dNdlnD_delegates(monkeypatch, small_binned_pop):
    called = {}

    def fake_compute(pop, name, cfg=None):
        called['args'] = (pop, name, cfg)
        return {"D": [1.0], "dNdlnD": [10.0]}

    # monkeypatch dispatcher.compute_variable which compute_plotdat will call
    import PyParticle.analysis.dispatcher as _disp
    monkeypatch.setattr(_disp, "compute_variable", fake_compute)
    out = compute_plotdat(small_binned_pop, "dNdlnD", {"N_bins": 5})
    assert isinstance(out, dict)
    assert called['args'][1] == "dNdlnD"
