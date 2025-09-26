from PyParticle.analysis import list_variables, describe_variable


def test_core_variables_present():
    names = list_variables()
    assert "dNdlnD" in names
    assert "Nccn" in names


def test_describe_variable_keys():
    meta = describe_variable("dNdlnD")
    assert "axis_keys" in meta
    assert "D" in meta["axis_keys"]
    assert "defaults" in meta
