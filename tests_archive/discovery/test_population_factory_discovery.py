import importlib, pkgutil
from PyParticle.population import factory as pop_factory
from PyParticle.population.builder import build_population

def iter_population_factory_modules():
    for m in pkgutil.iter_modules(pop_factory.__path__, pop_factory.__name__ + "."):
        yield m.name

def test_all_population_factory_modules_importable():
    for modname in iter_population_factory_modules():
        importlib.import_module(modname)

def test_build_population_unknown_type_raises():
    bad = {"type": "___nope___"}
    try:
        build_population(bad)
    except ValueError as e:
        assert "unknown" in str(e).lower() or "type" in str(e).lower()
    else:
        raise AssertionError("Expected ValueError for unknown type")
