# PyParticle Variable Computation (Simple Analysis Layer)

This directory implements the *simple/flat* variable computation pattern ("Option A") chosen for PyParticle.
It mirrors the existing population & optics factory style:

- One file per variable under `factory/` (e.g. `dNdlnD.py`, `Nccn.py`, `b_ext.py`).
- Each file defines a metadata-bearing class (subclass of `AbstractVariable`) and registers it with `@register_variable`.
- The registry discovers all variable modules lazily on first use.
- A thin dispatcher `compute_variable(population, varname, var_cfg=None)` unifies access.

## Public API
```python
from PyParticle.analysis_simple import compute_variable, list_variables, describe_variable

# List available canonical variable names
print(list_variables())  # e.g. ['Nccn', 'b_abs', 'b_ext', 'b_scat', 'dNdlnD', 'frac_ccn']

# Compute a size distribution
sd = compute_variable(pop, 'dNdlnD', {"N_bins": 100})
# sd -> {'D': array([...]), 'dNdlnD': array([...])}

# Compute extinction coefficients
ext = compute_variable(pop, 'b_ext', {"wvls": [450e-9, 550e-9], "rh_grid": [0.0, 0.5, 0.9]})
# ext -> {'rh_grid': array([...]), 'wvls': array([...]), 'b_ext': 2D array}
```

Aliases (deprecated) like `total_ext`, `total_abs`, `total_scat` resolve to canonical names with a warning.

## Adding a New Variable
1. Create a new file under `factory/`, e.g. `ssa.py`.
2. Implement a subclass:
```python
from ..base import AbstractVariable, VariableMeta
from ..registry import register_variable

@register_variable('ssa')
class SingleScatAlbedo(AbstractVariable):
    meta = VariableMeta(
        name='ssa',
        value_key='ssa',
        axis_keys=('rh_grid','wvls'),
        description='Single scattering albedo vs RH & wavelength',
        default_cfg={
            'wvls': [550e-9],
            'rh_grid': [0.0, 0.5],
            'morphology': 'core-shell',
            'T': 298.15,
        },
        units={'wvls': 'm', 'rh_grid': 'fraction', 'ssa': 'dimensionless'}
    )
    def compute(self, population):
        # derive using existing extinction/scattering variables OR optics builder
        ext = compute_variable(population, 'b_ext', self.cfg)  # recursion is allowed
        scat = compute_variable(population, 'b_scat', self.cfg)
        ssa = scat['b_scat'] / ext['b_ext']
        return {'rh_grid': ext['rh_grid'], 'wvls': ext['wvls'], 'ssa': ssa}
```
3. Done. No central switch edits required.

## Design Rationale
- Mirrors existing factory-discovery patterns (population, optics) for consistency.
- Keeps barrier to contribution low (single file addition).
- Dispatcher centralizes defaults merge, alias handling, and optional squeezing of trivial dimensions.

## Future Extensions
- Caching layer (population-attached or LRU) can be added inside dispatcher without changing variable modules.
- Introspection (`describe_variable`) supports UI generation or downstream systems (e.g., AMBRS) building dynamic forms.
- Additional axes (temperature, pressure) can be supported by extending `axis_keys` & compute logic.

## Migration Notes
Legacy `viz.data_prep` direct compute functions now emit `DeprecationWarning` and delegate to this dispatcher. Use `compute_variable` going forward for all programmatic access.
