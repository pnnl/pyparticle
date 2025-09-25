import json
import numpy as np
from PyParticle import build_population, build_optical_population
from PyParticle.analysis import dispatcher

cfg = json.load(open('examples/configs/binned_lognormal_bscat_wvl.json'))
base_pop_cfg = cfg['population']
var_cfg = cfg.get('b_scat', {})
print('var_cfg from json:', var_cfg)
pop = build_population(base_pop_cfg)
opt_cfg = dict(var_cfg or {})
morph = opt_cfg.get('morphology', 'core-shell')
opt_cfg.setdefault('type', str(morph).replace('-', '_'))
if 'wvl_grid' not in opt_cfg:
    opt_cfg['wvl_grid'] = opt_cfg.get('wvls', opt_cfg.get('wvl_grid', [550e-9]))
if 'rh_grid' not in opt_cfg:
    opt_cfg['rh_grid'] = opt_cfg.get('rh_grid', opt_cfg.get('rh_grid', [0.0]))
print('opt_cfg prepared:', opt_cfg)
opt_pop = build_optical_population(pop, opt_cfg)
vardat = dispatcher.compute_variable(opt_pop, 'b_scat', {'wvls': list(opt_cfg['wvl_grid']), 'rh_grid': list(opt_cfg['rh_grid'])})
print('vardat repr:')
for k,v in vardat.items():
    print('-', k, '->', repr(v))
    try:
        a = np.asarray(v)
        print('   shape=', a.shape, 'ndim=', a.ndim)
    except Exception as e:
        print('   could not array-ify:', e)
print('done')
