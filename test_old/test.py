# generic (marimo) nb testing
# i opened an issue at https://github.com/marimo-team/marimo/discussions/3683

from pathlib import Path
_ = Path(__file__).parent
_ = _ / 'data' / 'regression'
data = _
import pytest
@pytest.fixture(scope="session")
def datadir(): return data
@pytest.fixture(scope="session")
def original_datadir(): return data

@pytest.fixture
def nb():
    import binned_lognormal
    _ = binned_lognormal.app.run()
    _ = _[-1]   # results
    #_ = _._dict # regression testing expecting json but this has many pyparticle objects
    return _ 

def test(num_regression, nb):
    from itertools import product
    _ = product(['babs', 'bscat'],  ['single', 'two',] )
    _ = map(lambda ss: ss[0]+'_'+ss[1]+'AerosolSpecies', _)
    # only 1d arrays supported
    _ = {s:nb[s].flatten() for  s in _}
    num_regression.check(_)
