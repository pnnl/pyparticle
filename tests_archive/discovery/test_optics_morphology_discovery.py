from PyParticle.optics import factory as oreg

def test_homogeneous_available():
    types = oreg.registry._morphology_registry if hasattr(oreg, 'registry') else oreg.discover_morphology_types()
    # Prefer the public discover function
    if callable(getattr(oreg, 'discover_morphology_types', None)):
        types = oreg.discover_morphology_types()
    assert "homogeneous" in types
