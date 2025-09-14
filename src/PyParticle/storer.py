import numpy as np
from dataclasses import asdict
# from . import RI_fun


def get_output_filename(
        processed_output_dir, sample_id_str, run_num, timestep, 
        repeat_num=1, runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json'):
    output_filename = (
        processed_output_dir + sample_id_str + '_' + str(run_num).zfill(runid_digits) + '_' 
        + str(timestep).zfill(timestep_digits) + '_' + str(repeat_num).zfill(repeat_digits) 
        + '.' + file_format)
    
    return output_filename

def get_sample_padding(N_samples):
    return int(np.floor(np.log10(N_samples)) + 1)
    

def arrays_to_lists(dictionary):
    for one_key in dictionary.keys():
        if type(dictionary[one_key]) == type(np.zeros(2)):
            dictionary[one_key] = dictionary[one_key].tolist()
    return dictionary

def make_population_dictionary(optical_population):
    optical_pop_dict = {}
    
    optical_pop_dict['species'] = make_specs_dictionary(optical_population.species)
    optical_pop_dict['spec_masses'] = optical_population.spec_masses
    optical_pop_dict['num_concs'] = optical_population.num_concs
    optical_pop_dict['ids'] = optical_population.ids
    
    optical_pop_dict['rh_grid'] = optical_population.rh_grid
    optical_pop_dict['wvl_grid'] = optical_population.wvl_grid
    
    optical_pop_dict['temp'] = optical_population.temp
    
    optical_pop_dict['core_vols'] = optical_population.core_vols
    optical_pop_dict['shell_dry_vols'] = optical_population.shell_dry_vols
    optical_pop_dict['h2o_vols'] = optical_population.h2o_vols
    
    optical_pop_dict['core_ris'] = optical_population.core_ris
    optical_pop_dict['dry_shell_ris'] = optical_population.dry_shell_ris
    optical_pop_dict['h2o_ris'] = optical_population.h2o_ris
    
    optical_pop_dict['Cexts'] = optical_population.Cexts
    optical_pop_dict['Cscas'] = optical_population.Cscas
    optical_pop_dict['Cabss'] = optical_population.Cabss
    optical_pop_dict['gs'] = optical_population.gs
    optical_pop_dict['Cprs'] = optical_population.Cprs
    optical_pop_dict['Cbacks'] = optical_population.Cbacks
    
    optical_pop_dict['Cexts'] = optical_population.Cexts
    optical_pop_dict['Cscas'] = optical_population.Cscas
    optical_pop_dict['Cabss'] = optical_population.Cabss
    optical_pop_dict['gs'] = optical_population.gs
    optical_pop_dict['Cprs'] = optical_population.Cprs
    optical_pop_dict['Cbacks'] = optical_population.Cbacks
    
    optical_pop_dict['Cexts_bc'] = optical_population.Cexts_bc
    optical_pop_dict['Cscas_bc'] = optical_population.Cscas_bc
    optical_pop_dict['Cabss_bc'] = optical_population.Cabss_bc
    optical_pop_dict['gs_bc'] = optical_population.gs_bc
    optical_pop_dict['Cprs_bc'] = optical_population.Cprs_bc
    optical_pop_dict['Cbacks_bc'] = optical_population.Cbacks_bc

    optical_pop_dict['Cexts_clear'] = optical_population.Cexts_clear
    optical_pop_dict['Cscas_clear'] = optical_population.Cscas_clear
    optical_pop_dict['Cabss_clear'] = optical_population.Cabss_clear
    optical_pop_dict['gs_clear'] = optical_population.gs_clear
    optical_pop_dict['Cprs_clear'] = optical_population.Cprs_clear
    optical_pop_dict['Cbacks_clear'] = optical_population.Cbacks_clear
    
    # optical_pop_dict['Cexts_nobc'] = optical_population.Cexts_nobc
    # optical_pop_dict['Cscas_nobc'] = optical_population.Cscas_nobc
    # optical_pop_dict['Cabss_nobc'] = optical_population.Cabss_nobc
    # optical_pop_dict['gs_nobc'] = optical_population.gs_nobc
    # optical_pop_dict['Cprs_nobc'] = optical_population.Cprs_nobc
    # optical_pop_dict['Cbacks_nobc'] = optical_population.Cbacks_nobc
    
    return optical_pop_dict

def make_specs_dictionary(species,default_wvls=np.linspace(2e-7,7e-6,94)):
    aero_specs_dict = {}
    for one_spec in species:
        if type(one_spec.refractive_index.wvls) == type(None):
            wvls = default_wvls
        else:
            wvls = one_spec.refractive_index.wvls
        
        aero_specs_dict[one_spec.name] = asdict(one_spec)
        aero_specs_dict[one_spec.name]['refractive_index'] = asdict(one_spec.refractive_index)
        
        aero_specs_dict[one_spec.name]['refractive_index']['wvls'] = wvls
        if type(aero_specs_dict[one_spec.name]['refractive_index']['RI_params']) == type(None):
            aero_specs_dict[one_spec.name]['refractive_index']['real_ris'] = (
                aero_specs_dict[one_spec.name]['refractive_index']['real_ri_fun'](wvls))
            aero_specs_dict[one_spec.name]['refractive_index']['imag_ris'] = (
                aero_specs_dict[one_spec.name]['refractive_index']['imag_ri_fun'](wvls))
        else:
            RI_params = aero_specs_dict[one_spec.name]['refractive_index']['RI_params']
            n_550 = RI_params['n_550']
            alpha_n = RI_params['alpha_n']
            aero_specs_dict[one_spec.name]['refractive_index']['real_ris'] = (
                RI_fun(wvls, n_550, alpha_n))
            k_550 = RI_params['k_550']
            alpha_k = RI_params['alpha_k']
            aero_specs_dict[one_spec.name]['refractive_index']['imag_ris'] = (
                RI_fun(wvls, k_550, alpha_k))
        
        del aero_specs_dict[one_spec.name]['refractive_index']['real_ri_fun']
        del aero_specs_dict[one_spec.name]['refractive_index']['imag_ri_fun']
            
        aero_specs_dict[one_spec.name] = arrays_to_lists(aero_specs_dict[one_spec.name])
        aero_specs_dict[one_spec.name]['refractive_index'] = arrays_to_lists(
            aero_specs_dict[one_spec.name]['refractive_index'])
    
    return aero_specs_dict
    
def separate_ris(optical_pop_dict):
    optical_pop_dict['core_ris_real'] = np.real(optical_pop_dict['core_ris'])
    optical_pop_dict['core_ris_imag'] = np.imag(optical_pop_dict['core_ris'])
    
    optical_pop_dict['dry_shell_ris_real'] = np.real(optical_pop_dict['dry_shell_ris'])
    optical_pop_dict['dry_shell_ris_imag'] = np.imag(optical_pop_dict['dry_shell_ris'])
    
    optical_pop_dict['h2o_ris_real'] = np.real(optical_pop_dict['h2o_ris'])
    optical_pop_dict['h2o_ris_imag'] = np.imag(optical_pop_dict['h2o_ris'])
    
    del optical_pop_dict['core_ris']
    del optical_pop_dict['dry_shell_ris']
    del optical_pop_dict['h2o_ris']
    
    return optical_pop_dict
