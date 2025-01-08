import PyParticle
import json
import numpy as np
from importlib import reload
reload(np)
partmc_dir = '/Users/fier887/Downloads/box_simulations3/library_18_abs2/0003/'
run_num = 3
timestep = 18
repeat_num = 1
particle_population = PyParticle.builder.partmc.build({'partmc_dir': '/Users/fier887/Downloads/box_simulations3/library_18_abs2/0003/', 'timestep': 18, 'repeat': 1},n_particles=None, species_modifications={'SOA': {'k_550': 0.001, 'alpha_k': 0}})
optical_population = PyParticle.make_optical_population(particle_population, np.array([0.0,0.99]), np.array([5.5e-07,7.5e-07]), species_modifications={'SOA': {'k_550': 0.001, 'alpha_k': 0}})
optical_pop_dict = PyParticle.make_population_dictionary(optical_population)
optical_pop_dict = PyParticle.separate_ris(optical_pop_dict)
optical_pop_dict = PyParticle.arrays_to_lists(optical_pop_dict)
optical_pop_dict['population_settings'] = {'partmc_dir': '/Users/fier887/Downloads/box_simulations3/library_18_abs2/0003/', 'timestep': 18, 'repeat': 1}
optical_pop_dict['species_modifications'] = {'SOA': {'k_550': 0.001, 'alpha_k': 0}}
output_filename =  'sample.json'
output_filename = PyParticle.get_output_filename('processed_output_dir', run_num, timestep, repeat_num=repeat_num, runid_digits=4)
with open(output_filename, "w") as outfile:
    json.dump(optical_pop_dict, outfile)
