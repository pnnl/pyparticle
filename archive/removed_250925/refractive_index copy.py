import numpy as np
from .. import data_path
from  ..utilities import get_number

from typing import Callable, Optional
from dataclasses import dataclass

@dataclass
class RefractiveIndex:
    """RefractiveIndex: the definition of wavelength-dependent refractive index.
    Applies to an individual aerosol component or for a mixture of components """
    
    # convention is to always use real_ri_fun and imag_ri_fun if they are defined
    # (i.e., wvls, real_ris, and imag_ris are ignored)
    real_ri_fun: Callable[[float],float]
    imag_ri_fun: Callable[[float],float]
    
    # if real_ris and imag_ris are defined, wvls must also be defined
    wvls: Optional[np.array] = None
    real_ris: Optional[np.array] = None
    imag_ris: Optional[np.array] = None
    
    RI_params: Optional[dict] = None
    

# class RefractiveIndex:
#     """
#     Example implementation of a wavelength-dependent refractive index.
#     """
#     def __init__(self, wavelengths, n_values, k_values=None):
#         self.wavelengths = np.array(wavelengths)
#         self.n = np.array(n_values)
#         if k_values is not None:
#             self.k = np.array(k_values)
#         else:
#             self.k = np.zeros_like(self.n)
    
#     # fixme: check this
#     def __call__(self, wavelength):
#         # Interpolate to get n and k at the specified wavelength(s)
#         n_interp = np.interp(wavelength, self.wavelengths, self.n)
#         k_interp = np.interp(wavelength, self.wavelengths, self.k)
#         return n_interp + 1j * k_interp


def RI_fun(species, wvl_grid, temp=293.15, specdata_path=data_path / 'species_data', species_modifications={}):
    """
    Function to retrieve the refractive index for a given species and wavelength grid.
    """
    if species.name.upper()!='H2O':
        RI_param_dict = get_RI_params(species.name.upper())
        wvl_dep_fun = lambda wvl, val_550, val_alpha: val_550*(wvl/550e-9)**val_alpha
        
        real_ri_fun = lambda wvl: RI_param_dict['n_550']*(wvl/550e-9)**RI_param_dict['alpha_n']
        imag_ri_fun = lambda wvl: RI_param_dict['k_550']*(wvl/550e-9)**RI_param_dict['alpha_k']
    else: # water has tabulated n and k values
        RI_datafile = specdata_path / 'ri_water.csv'
        wvl_micron, wavenumber, freq, ns, ks, alpha = np.genfromtxt(RI_datafile, delimiter=",", skip_header=1, unpack=True)        
        real_ri_fun = lambda wvl: np.interp(wvl, wvl_micron*1e-6, ns)
        imag_ri_fun = lambda wvl: np.interp(wvl, wvl_micron*1e-6, ks)
        
         

    return RefractiveIndex(real_ri_fun=real_ri_fun, imag_ri_fun=imag_ri_fun,
                         wvls=wvl_grid, real_ris=real_ri_fun(wvl_grid), imag_ris=imag_ri_fun(wvl_grid))
    #RefractiveIndex(wvl_grid, n, k)


    # if species.name!='H2O':
    #     RI_datafile = specdata_path / 'aero_RI_data.dat'
    #     with open(RI_datafile) as data_file:
    #         for line in data_file:
    #             if line.startswith(species.name):
    #                 name_in_file, n_550, k_550, alpha_n, alpha_k = line.split()
    #                 if species.name in species_modifications.keys():
    #                     if 'n_550' in species_modifications[species.name].keys():
    #                         n_550=species_modifications[species.name]['n_550']
    #                     if 'k_550' in species_modifications[species.name].keys():
    #                         k_550=species_modifications[species.name]['k_550']
    #                     if 'alpha_n' in species_modifications[species.name].keys():
    #                         alpha_n=species_modifications[species.name]['alpha_n']
    #                     if 'alpha_k' in species_modifications[species.name].keys():
    #                         alpha_k=species_modifications[species.name]['alpha_k']
    #                 n=float(n_550)*((1e9*wvl_grid)/550)**float(alpha_n)
    #                 k=float(k_550)*((1e9*wvl_grid)/550)**float(alpha_k)
    #                 return RefractiveIndex(wvl_grid, n, k)
    # else:
    #     RI_datafile = specdata_path / 'ri_water.csv'
    #     wvl_micron, wavenumber, freq, ns, ks, alpha = np.genfromtxt(RI_datafile, delimiter=",", skip_header=1, unpack=True)        
    #     k = np.interp(wvl_grid*1e6, wvl_micron, ks)
    #     n = np.interp(wvl_grid*1e6, wvl_micron, ns)
    #     return RefractiveIndex(wvl_grid, n, k)

# def RI_fun(wvl, val_550, val_alpha):
#     return val_550*(wvl/550e-9)**val_alpha

def add_RI_to_spec(
        aero_spec,wvls=None,
        specdata_path=data_path / 'species_data',
        species_modifications={}):
    
    spec_name = aero_spec.name.upper()
    aero_spec.refractive_index =  RI_fun(aero_spec, wvls, temp=None, specdata_path=specdata_path, species_modifications=species_modifications)
    
    # if spec_name != 'H2O':
    #     RI_params = get_RI_params(spec_name)
    #     if 'n_550' in species_modifications.keys():
    #         RI_params['n_550'] = species_modifications['n_550']
        
    #     if 'alpha_n' in species_modifications.keys():
    #         RI_params['alpha_n'] = species_modifications['alpha_n']
        
    #     real_ri_fun = lambda wvl: RI_fun(wvl, RI_params['n_550'], RI_params['alpha_n'])
        
    #     if 'k_550' in species_modifications.keys():
    #         RI_params['k_550'] = species_modifications['k_550']
        
    #     if 'alpha_k' in species_modifications.keys():
    #         RI_params['alpha_k'] = species_modifications['alpha_k']
        
    #     imag_ri_fun = lambda wvl: RI_fun(wvl, RI_params['k_550'], RI_params['alpha_k'])
        
    #     if type(wvls) == type(None):
    #         real_ris = None
    #         imag_ris = None
    #     else:
    #         real_ris = real_ri_fun(wvls)
    #         imag_ris = imag_ri_fun(wvls)
    # else:
    #     ri_h2o_filename = specdata_path / 'ri_water.csv'
    #     wavelength_list = []
    #     n_list = []
    #     k_list = []
    #     with open(ri_h2o_filename) as data_file:
    #         for line in data_file:
    #             if not 'Wavelength' in line:
    #                 split_output = line.split(',')
    #                 wavelength_list.append(1e-6*get_number(split_output[0]))
    #                 n_list.append(get_number(split_output[3]))
    #                 k_list.append(get_number(split_output[4]))
    #         wvls = np.hstack(wavelength_list)
    #         real_ris = np.hstack(n_list)
    #         imag_ris = np.hstack(k_list)
        
    #     real_ri_fun = interpolate.interp1d(wvls, real_ris)# lambda wvl: interpolate.interp1d(wvls, real_ris)(wvl)
    #     imag_ri_fun = interpolate.interp1d(wvls, imag_ris) # lambda wvl: interpolate.interp1d(wvls, imag_ris)(wvl)#np.interp(wvl, wvls, imag_ris)  
    #     RI_params = None
    
    # aero_spec.refractive_index = RefractiveIndex(
    #     wvls = wvls,
    #     real_ris = real_ris,
    #     imag_ris = imag_ris,
    #     real_ri_fun = real_ri_fun,
    #     imag_ri_fun = imag_ri_fun,
    #     RI_params = RI_params)
    
    return aero_spec

def get_RI_params(name):
    if name.upper() in ['SO4','NH4','NO3','NA','CL','MSA','CO3']:
        k_550 = 0.
        n_550 = 1.55
        alpha_n = 0.044 
        alpha_k = 0.
        
        # based on wavelength dependence of NaNO3 (only inorganic salt at RH=0%)
        # data from here: https://eodg.atm.ox.ac.uk/ARIA/data?Salts/Sodium_Nitrate/10%25_(Cotterell_et_al._2017)/NaNO3_10_Cotterell_2017.ri
        # underlying data:
        #   Reference: Cotterell, M.I., Willoughby, R.E., Bzdek, B.R., Orr-Ewing, A.J. and Reid, J.P., A Complete Parameterization of the Relative Humidity and Wavelength Dependence of the Refractive Index of Hygroscopic Inorganic Aerosol Particles.
        #   DOI: 10.5194/acp-17-9837-2017
    elif name.upper() == 'BC':
        k_550 = 0.74
        n_550 = 1.82
        alpha_n = 0.
        alpha_k = 0.
    elif name.upper() == 'OIN':
        k_550 = 0.006
        n_550 = 1.68
        alpha_n = 0.
        alpha_k = 0.
    else: # organics
        k_550 = 0.
        n_550 = 1.45
        alpha_n = 0.
        alpha_k = 0.
        
        #warn(name + ' is not in the list, assuming its an arbitrary organic')
    
    RI_param_dict = {'n_550':n_550, 'k_550':k_550, 'alpha_n':alpha_n, 'alpha_k':alpha_k}
    
    return RI_param_dict