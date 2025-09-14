# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
"""
# PyParticle python package

# @author: Laura Fierce
"""
# import os
# import numpy as np
# from pathlib import Path
# ### TODO: this should be an argument for a process
# try: # 'dev' situation
#     laura_datapath = '/Users/fier887/OneDrive - PNNL/Code/PyParticle/datasets'
#     if os.path.exists(laura_datapath):
#         data_path = Path(laura_datapath)
#     else:
#         from pyprojroot import here
#         data_path = here() / 'datasets'
# except ImportError:
#     raise FileNotFoundError('data_path not set')
# ### 
# from .utilities import get_number
# # #from .utilities import Py3Wrapper
# from .aerosol_particle import Particle, make_particle, make_particle_from_masses
# from .aerosol_species import AerosolSpecies, retrieve_one_species
# from .particle_population import ParticlePopulation

# from .optics import make_optical_particle, make_optical_population
# from .optics import RefractiveIndex, RI_fun
# #from .optics import CoreShellParticle, CoreShellPopulation

# from .storer import ( 
#     arrays_to_lists, make_population_dictionary, make_specs_dictionary, 
#     separate_ris, get_output_filename, get_sample_padding)

# from . import builder
# from .builder import monodisperse
# # from .builder import partmc

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyParticle python package

@author: Laura Fierce
"""
import os
import numpy as np
from pathlib import Path


### TODO: this should be an argument for a process
try:  # 'dev' situation
    laura_datapath = '/Users/fier887/OneDrive - PNNL/Code/PyParticle/datasets'
    if os.path.exists(laura_datapath):
        data_path = Path(laura_datapath)
    else:
        from pyprojroot import here
        data_path = here() / 'datasets'
except ImportError:
    raise FileNotFoundError('data_path not set')

from .utilities import get_number
# #from .utilities import Py3Wrapper

from .aerosol_particle import Particle, make_particle, make_particle_from_masses

# Updated imports for new species/registry structure
from .species.base import AerosolSpecies
from .species.registry import (
    get_species,
    register_species,
    list_species,
    extend_species,
    retrieve_one_species,
)

from .population.base import ParticlePopulation
from .population import build_population

from .optics.builder import build_optical_particle, build_optical_population
# from .optics import make_optical_particle, make_optical_population
# from .optics import RefractiveIndex, RI_fun
# from .optics import CoreShellParticle, CoreShellPopulation

from .storer import (
    arrays_to_lists, make_population_dictionary, make_specs_dictionary,
    separate_ris, get_output_filename, get_sample_padding
)

# # Optionally import builder submodule and monodisperse population creator
# from . import builder
# from .builder import monodisperse
# # from .builder import partmc