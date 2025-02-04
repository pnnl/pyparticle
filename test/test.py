# generic (marimo) nb testing
# i opened an issue at https://github.com/marimo-team/marimo/discussions/3683

from pathlib import Path
_ = Path(__file__).parent
_ = _ / 'regression'
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
    #                                                                only 1d arrays supported
    _ = {'babs_singleAerosolSpecies':  nb['babs_singleAerosolSpecies'].flatten() }
    num_regression.check(_)
