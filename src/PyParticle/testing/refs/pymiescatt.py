import numpy as np

def run(cfg: dict) -> dict[str, np.ndarray]:
    """
    Adapter for side-by-side comparisons.
    Given the full scenario cfg, build a comparable reference using PyMieScatt-based
    helper(s) available in the repo and return arrays keyed by 'b_scat','b_abs','b_ext'.
    Ensure wavelengths are in meters; return 1D arrays aligned with cfg['optics']['wvl_grid'].
    """
    # Pseudocode here; fill in using your existing helper functions/utilities:
    #  - extract population (cfg['population']) and grids (cfg['optics'])
    #  - compute b_scat/b_abs/b_ext with PyMieScatt reference routine
    #  - return dict with 1D arrays
    raise NotImplementedError("Fill in using your repo's PyMieScatt utilities.")
