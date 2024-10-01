import PyParticle
from PyMieScatt.CoreShell import MieQCoreShell

Dcore = 1e-7
Dshell = 1e-6
RH = 0.
wvl = 550e-9
m_core = 1.82 + 0.74j # assume core is BC
m_shell = 1.55 # assume shell is SO4

# step 1: compute optical properties of particle from PyMieScatt. This is your benchmark.

# JOSCELYNE -- add code here

# step 2: compute the things you need to make a particle.

# define the species for "BC" and "SO4". Use "retrieve_one_species" from PyParticle.aerosol_species
# compute the mass fraction of BC and mass fraction of SO4. Use the "density" from the species classes

# JOSCELYNE -- add code here

# step 3: make a particle PyParticle
# JOSCELYNE -- modify the code below

#aero_spec_names = ['BC','SO4','H2O']
#aero_spec_fracs = np.hstack([?,?,?])
particle = PyParticle.make_particle(D_shell, aero_spec_names, aero_spec_fracs)

# step 3: make an optical particle PyParticle
# JOSCELYNE -- modify the code below

rh_grid = np.hstack([RH])
wvl_grid = np.hstack([550e-9])

cs_particle = PyParticle.make_optical_particle(particle, rh_grid, wvl_grid)

