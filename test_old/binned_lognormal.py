import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def ex1():
    import numpy as np
    import PyParticle

    wvl_grid = np.linspace(350e-9,1150e-9,20) 
    rh_grid = [0.] 

    # =============================================================================
    # Example 1: lognormal distribution of with single dry aerosol species.
    #   Will lkely want to set rh_grid=[0.] for comparison with PyMieScatt
    # =============================================================================

    # example: distribution that includes a single aerosol component
    # k: imaginary part of the refractive index (RI)
    # n: real part of the RI
    # alpha_k: angstrom exponent for the imaginary part of the RI (describes variation with wavelength)
    # alpha_n: angstrom exponent for the real part of the RI

    # diameter grid varies logarithmically from D_min to D_max
    # N_bins specifies number of bins used to descritize the size distribution
    # GMD = geometric mean diameter
    # GSD = geometric standard deviation 
    single_population_settings = {
        'D_min':1e-9,
        'D_max':1e-6, 
        'N_bins':100,
        'Ntot':1e9,
        'GMD':1e-7,
        'GSD':1.6,
        'aero_spec_names':['BC'], # one aerosol species, organic carbon (OC)
        'aero_spec_fracs':np.array([1.])}
    population_singleAerosolSpecies = PyParticle.builder.binned_lognormal.build(
        single_population_settings,
        species_modifications={}, # when species_modifications is an empty dictionary, defaults are used
        D_is_wet=True) # if False, water content is not specified in the particle. Will be added.

    # When computing optical properties, ach particle represented as a homogeneous sphere if no core species are included.
    # Refractive index of spherical aprticle is volume-weighted average of RIs for OC and H2O
    optical_population_singleAerosolSpecies = PyParticle.make_optical_population(
        population_singleAerosolSpecies, rh_grid, wvl_grid,
        morphology='core-shell',compute_optics=True,temp=293.15,
        species_modifications={})

    # absorption and scattering coefficients at each wavelength
    babs_singleAerosolSpecies = optical_population_singleAerosolSpecies.get_optical_coeff('total_abs')
    bscat_singleAerosolSpecies = optical_population_singleAerosolSpecies.get_optical_coeff('total_scat')
    return (
        PyParticle,
        babs_singleAerosolSpecies,
        bscat_singleAerosolSpecies,
        np,
        optical_population_singleAerosolSpecies,
        population_singleAerosolSpecies,
        rh_grid,
        single_population_settings,
        wvl_grid,
    )


@app.cell
def ex2(PyParticle, np, rh_grid, wvl_grid):
    # =============================================================================
    # Example 2: lognormal distribution with two components
    # - assume all particles contain the same mass fraction of dry aerosol species
    # 
    # =============================================================================
    two_population_settings = {
        'D_min':1e-9,
        'D_max':1e-6,
        'N_bins':100,
        'Ntot':1e9,
        'GMD':1e-7,
        'GSD':1.6,
        'aero_spec_names':['BC','OC'], # two species, black carbon (BC) and organic carbon (OC)
        'aero_spec_fracs':np.array([0.1,0.9])}

    species_modifications = {
        'BC':{'k_550':0.7,'alpha_k':0,'n_550':1.7,'alpha_n':0},
        'OC':{'k_550':0,'alpha_k':0,'n_550':1.6,'alpha_n':0}}

    # Each particle represented as a concentric core-shell structure. (Will deal with more complex gemeotries down the line)
    # BC treated as an insoluble core at the center of the particle
    # other components are assumed to be a well-mixed shell

    population_twoAerosolSpecies = PyParticle.builder.binned_lognormal.build(
        two_population_settings, 
        species_modifications=species_modifications, # setting species_modifications modifies the properties as different from the default
        D_is_wet=False)
    optical_population_twoAerosolSpecies = PyParticle.make_optical_population(
        population_twoAerosolSpecies, rh_grid, wvl_grid,
        morphology='core-shell',compute_optics=True,temp=293.15,
        species_modifications=species_modifications)

    # absorption and scattering coefficients at each wavelength
    babs_twoAerosolSpecies = optical_population_twoAerosolSpecies.get_optical_coeff('total_abs')
    bscat_twoAerosolSpecies = optical_population_twoAerosolSpecies.get_optical_coeff('total_scat')
    return (
        babs_twoAerosolSpecies,
        bscat_twoAerosolSpecies,
        optical_population_twoAerosolSpecies,
        population_twoAerosolSpecies,
        species_modifications,
        two_population_settings,
    )


@app.cell
def plt(
    babs_singleAerosolSpecies,
    babs_twoAerosolSpecies,
    bscat_singleAerosolSpecies,
    bscat_twoAerosolSpecies,
    wvl_grid,
):
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(2,1,sharex=True)
    axs[0].plot(wvl_grid, babs_singleAerosolSpecies.transpose(), label='single species')
    axs[0].plot(wvl_grid, babs_twoAerosolSpecies.transpose(), label='two species')

    axs[1].plot(wvl_grid, bscat_singleAerosolSpecies.transpose(), label='single species')
    axs[1].plot(wvl_grid, bscat_twoAerosolSpecies.transpose(), label='two species')

    axs[1].set_xlabel('wavelength [m]')
    axs[0].set_ylabel('absorption\ncoeff. [m$^{-1}$]')
    axs[1].set_ylabel('scatttering\ncoeff. [m$^{-1}$]')
    axs[0].legend()

    axs[0].set_xlim([min(wvl_grid),max(wvl_grid)])
    axs[0]
    return axs, fig, plt



if __name__ == "__main__":
    app.run()
