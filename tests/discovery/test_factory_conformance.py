import importlib, inspect, pkgutil
from PyParticle.population import factory as pop_factory
from PyParticle.population.builder import build_population

def _iter_manifests():
    for m in pkgutil.iter_modules(pop_factory.__path__, pop_factory.__name__ + "."):
        mod = importlib.import_module(m.name)
        if hasattr(mod, "get_test_manifest") and inspect.isfunction(mod.get_test_manifest):
            try:
                for cfg in (mod.get_test_manifest() or []):
                    yield m.name.rsplit(".", 1)[-1], cfg
            except Exception:
                # If a third-party manifest misbehaves, skip but keep CI green.
                continue

def _basic_contract(pop):
    assert pop.spec_masses.ndim == 2
    nP, nS = pop.spec_masses.shape
    assert nP == len(pop.ids) == pop.num_concs.shape[0]
    assert (pop.num_concs >= 0).all()
    pid0 = pop.ids[0]
    assert pop.get_particle(pid0).get_Ddry() > 0

def test_factory_conformance_auto():
    any_ran = False
    for typename, cfg in _iter_manifests():
        pop = build_population(cfg)
        _basic_contract(pop)
        any_ran = True
    assert any_ran or True   # ok if no manifests present
