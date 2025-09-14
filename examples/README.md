# Examples

Purpose: Demonstrate building populations and computing optics via JSON configs.

Scripts

- `homogeneous_binned_lognormal.py` — binned lognormal population with homogeneous optics
- `core_shell_binned_lognormal.py` — binned lognormal population with core-shell optics

Configs

- `examples/configs/*.json` — population and optics configuration files (wavelengths often given in microns; the scripts convert to meters for the builder).

Run

- By default, tests gate examples with `PYPARTICLE_RUN_EXAMPLES=0`. To run manually:

```bash
python examples/homogeneous_binned_lognormal.py
python examples/core_shell_binned_lognormal.py
```
