"""PyParticle.analysis - functions and tools for analyzing particle populations.
"""
import numpy as np
import scipy.stats  # for KDE
from .optics import build_optical_population

def compute_dNdlnD(
        particle_population, wetsize=True, normalize=False,
        method='hist', N_bins=30,
        D_min=1e-9, D_max=1e-4,
        diam_scale='log'):
    """Compute size distribution (dN/dlnD) for the particle population.

    Returns a dict with:
        - 'D': bin-centers (meters)
        - 'dNdlnD': values for each bin (same length as centers)

    Notes:
    - Uses histogram over log-spaced diameter bins when method=='hist'.
    - Weights are taken from particle_population.num_concs unless normalize=True.
    """
    if diam_scale == 'log':
        # create log-spaced bin edges (meters)
        edges = np.logspace(np.log10(D_min), np.log10(D_max), N_bins + 1)
    else:
        raise NotImplementedError(f"diam_scale={diam_scale} not yet implemented")   
    
    # pick weights per particle id
    try:
        num_concs = np.asarray(particle_population.num_concs, dtype=float)
    except Exception:
        # fallback: assume equal weighting
        num_concs = None

    # gather diameters
    Ds = []
    weights = []
    for idx, part_id in enumerate(particle_population.ids):
        particle = particle_population.get_particle(part_id)
        d = particle.get_Dwet() if wetsize else particle.get_Ddry()
        Ds.append(d)
        if num_concs is not None and idx < len(num_concs):
            weights.append(num_concs[idx])
        else:
            weights.append(1.0)

    Ds = np.asarray(Ds, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if normalize:
        if weights.sum() > 0:
            weights = weights / weights.sum()

    if method == 'hist':
        hist, _ = np.histogram(Ds, bins=edges, weights=weights)
        # convert histogram to dN/dlnD: divide by bin width in ln space
        dln = np.log(edges[1:]) - np.log(edges[:-1])
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            dNdlnD = np.where(dln > 0, hist / dln, 0.0)
    elif method == 'kde':
        # KDE on log(D)
        logD = np.log(Ds)
        kde = scipy.stats.gaussian_kde(logD, weights=weights, bw_method='scott')
        centers = np.sqrt(edges[:-1] * edges[1:])
        dNdlnD = kde.evaluate(np.log(centers))
    else:
        raise NotImplementedError(f"method={method} not yet implemented")

    centers = np.sqrt(edges[:-1] * edges[1:])
    size_dict = {'D': centers, 'dNdlnD': dNdlnD}
    return size_dict

def compute_Nccn(
        particle_population, s_eval: np.ndarray, T: float) -> np.ndarray:
    """
    Compute cloud condensation nuclei (CCN) spectrum for the particle population.
    s_eval: array of environmental supersaturation values.
    normalize: if False, compute total number concentration of CCN; if True, compute fraction.
    Returns: CCN spectrum as array (number concentration or fraction at each supersaturation).
    """
    
    ccn_dict = {}
    ccn_spectrum = np.zeros_like(s_eval, dtype=float)
    # safe default temperature
    for idx, s_env in enumerate(s_eval):
        ccn_count = 0.0
        for i, part_id in enumerate(particle_population.ids):
            particle = particle_population.get_particle(part_id)
            s_crit = particle.get_critical_supersaturation(T, return_D_crit=False)
            num_conc = float(particle_population.num_concs[i])
            if s_env >= s_crit:
                ccn_count += num_conc
        ccn_spectrum[idx] = ccn_count
    
    ccn_dict['s'] = s_eval
    ccn_dict['Nccn'] = ccn_spectrum
    return ccn_dict

def compute_optical_coeffs(
        self, 
        coeff_types:list=['total_scat','total_abs'],
        wvls:np.ndarray=np.array([550e-9]),
        rh_grid:np.ndarray=None, # fixme: this seems like a janky way to make "model RH" the default
        morphology:str='core-shell', 
        # fixme: recalculate, even if already in "diagnostics"
        ) -> np.ndarray:
    """
    Compute optical coefficients for the particle population at specified wavelengths and humidities.
    wvls: list of wavelengths in meters.
    rh_grid: list of RH values (0 to <1 in this case).
    optics_type: 'total_abs', 'pure_bc_abs', 'clear_abs', etc.
    bconly: if True, only sum particles containing black carbon.
    species_modifications: dictionary to override default species optical properties.
    Returns: dict with keys for absorption, scattering, etc.
    """
    
    optical_population = build_optical_population(
        self.particle_population, rh_grid, wvls,
        morphology=morphology, compute_optics=True, temp=temp,
        species_modifications=self.species_modifications
    )
    optical_coeff_dict = {'wvls': wvls, 'rh_grid': rh_grid}

    for coeff_type in coeff_types:
        optical_coeff_dict[coeff_type] = optical_population.get_optical_coeff(
            optics_type=coeff_type,
            rh='all', wvl='all',
            bconly=False)

    return optical_coeff_dict

def compute_variable(particle_population,varname,var_cfg,return_plotdat=False):
    if varname == 'dNdlnD':
        diam_scale = var_cfg.get('diam_scale','log')
        vardat = compute_dNdlnD(
            particle_population,
            wetsize = var_cfg.get('wetsize',True),
            normalize = var_cfg.get('wetsize',False),
            method = var_cfg.get('method','hist'),
            N_bins = var_cfg.get('N_bins',30),
            D_min = var_cfg.get('D_min',1e-9),
            D_max = var_cfg.get('D_max',1e-4),
            diam_scale = diam_scale)
        y = vardat['dNdlnD']
        if diam_scale == 'log':
            x = np.logspace(np.log10(var_cfg.get('D_min',1e-9)), np.log10(var_cfg.get('D_max',1e-4)), var_cfg.get('N_bins',30))
            labs = ['D (m)', 'dN/dlnD (1/m$^3$)']
        else:
            raise NotImplementedError(f"diam_scale={diam_scale} not yet implemented")
    elif varname == 'Nccn':
        vardat = compute_Nccn(particle_population,var_cfg['s_eval'], var_cfg['T'])
        y = vardat['Nccn']
        x = var_cfg['s_eval']
        labs = ['s (%)', 'Nccn (1/m$^3$)']
    elif varname == 'frac_ccn':
        vardat = compute_Nccn(particle_population,var_cfg['s_eval'], var_cfg['T'])
        # return fraction
        total = np.sum(particle_population.num_concs)
        if total > 0:
            vardat['Nccn'] = vardat['Nccn'] / total
        y = vardat['Nccn']
        x = var_cfg['s_eval']
        labs = ['s (%)', 'fraction CCN']
    elif varname in ['b_abs','b_scat','b_ext','total_abs','total_scat','total_ext']:
        vardat = compute_optical_coeffs(particle_population,var_cfg)
        if varname in ['b_abs','total_abs']:
            varname_reformatted = 'abs. coeff.'
        elif varname in ['b_scat','total_scat']:
            varname_reformatted = 'scat. coeff.'
        elif varname in ['b_ext','total_ext']:
            varname_reformatted = 'ext. coeff.'
        else:
            varname_reformatted = varname
        
        y = vardat[varname]
        if var_cfg.get('vs_rh',False):
            x = var_cfg['rh_grid']
            labs = ['RH (%)', varname_reformatted + ' (1/m)']
        elif var_cfg.get('vs_wvl',True):
            x = var_cfg['wvls']
            labs = ['wavelength (m)', varname_reformatted + ' (1/m)']
        else:
            raise NotImplementedError("optical coeffs only implemented vs RH or wavelength")
    # elif varname == 'b_abs':
    #     var_cfg['name'] = 'total_abs'
    #     vardat = compute_optical_coeffs(particle_population,var_cfg)
    # elif varname == 'b_scat':
    #     var_cfg['name'] = 'total_ascat'
    #     vardat = compute_optical_coeffs(particle_population,var_cfg)
    # elif varname == 'b_ext':
    #     var_cfg['name'] = 'total_ext'
    #     vardat = compute_optical_coeffs(particle_population,var_cfg)
    elif varname == 'Ntot':
        vardat = np.sum(particle_population.num_concs)
        x = None
        labs = ['Ntot (1/m$^3$)']
    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")
    
    if return_plotdat:
        return x, y, labs
    else:
        return vardat

def build_default_var_cfg(varname):
    if varname == 'dNdlnD':
        var_cfg = {
            'wetsize': True,
            'normalize': False,
            'method': 'hist',
            'N_bins': 30,
            'D_min': 1e-9,
            'D_max': 1e-4,
            'diam_scale': 'log'
        }
    elif varname == 'Nccn':
        var_cfg = {
            's_eval': np.linspace(0.01,1.0,50), # 0.01% to 1%
            'T': 298.15 # K
        }
    elif varname == 'frac_ccn':
        var_cfg = {
            's_eval': np.linspace(0.01,1.0,50), # 0.01% to 1%
            'T': 298.15 # K
        }
    elif varname in ['b_abs','b_scat','b_ext','total_abs','total_scat','total_ext']:
        var_cfg = {
            'wvls': np.array([550e-9]), # meters
            'rh_grid': np.array([0.0,0.5,0.7,0.85,0.9,0.95,0.98,0.99]), # RH values
            'morphology': 'core-shell',
            'vs_wvl': True,
            'vs_rh': False
        }
    elif varname == 'Ntot':
        var_cfg = {}
    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")
    
    return var_cfg  