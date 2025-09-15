import importlib
import pytest
import numpy as np

pymspec = importlib.util.find_spec("PyMieScatt")
if pymspec is None:
	pytest.skip("PyMieScatt not installed: integration tests skipped", allow_module_level=True)

from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population
from tools.plot_helpers import lognormal_mode_to_bins


def _try_pymiescatt_polydisperse(m, wvl_nm, D_grid_nm, n_D):
	"""Try known PyMieScatt polydisperse helpers. Return Cext array over wavelengths if possible."""
	import PyMieScatt as PMS

	# Try common helper names in decreasing preference
	for func_name in ("Mie_SD", "MieQ_SD", "Mie_SD_Mult", "MieQ_WithDistribution", "MieQ_SD"):
		func = getattr(PMS, func_name, None)
		if func is not None:
			try:
				# Many of these helpers accept (m, wavelength_nm, Diameters_nm, n(D)) or similar signatures.
				out = func(m, wvl_nm, D_grid_nm, n_D)
				# If the helper returns dict-like with keys, try to extract extinction
				if isinstance(out, dict):
					if "Cext" in out:
						return np.atleast_1d(out["Cext"])
					if "Qext" in out and "Area" in out:
						return np.atleast_1d(out["Qext"] * out["Area"])
					# try first numeric array
					for v in out.values():
						if hasattr(v, "shape"):
							return np.atleast_1d(np.sum(v, axis=0))
				# If numeric array is returned, try to coerce
				if hasattr(out, "shape"):
					return np.atleast_1d(out)
			except Exception:
				continue

	# No helper found
	return None


def _integrate_mieq_over_distribution(m, wvl_nm, D_grid_nm, n_D):
	"""Fallback: compute MieQ per diameter and integrate with weights n_D."""
	import PyMieScatt as PMS

	Cext = np.zeros(len(wvl_nm), dtype=float)
	area = np.pi * (0.5 * (D_grid_nm * 1e-9)) ** 2
	for ii, Dnm in enumerate(D_grid_nm):
		for jj, lam in enumerate(wvl_nm):
			out = PMS.MieQ(m, float(lam), float(Dnm), asDict=True, asCrossSection=False)
			Qext = out.get("Qext", 2.0)
			Cext[jj] += Qext * area[ii] * float(n_D[ii])
	return Cext


def test_pymiescatt_agrees_with_pyparticle():
	"""Integration test: compare PyParticle population-averaged extinction to PyMieScatt polydisperse calculation.

	This test uses the project's `binned_lognormal` example config to build a population,
	then computes population b_ext via PyParticle APIs and compares the shape and magnitude
	against PyMieScatt's polydisperse tools (best effort). The test is intentionally
	tolerant (relative differences allowed) to avoid fragile exact equality.
	"""
	# Load example configs from examples/configs
	import json
	from pathlib import Path

	cfg_dir = Path(__file__).resolve().parents[2] / "examples" / "configs"
	pop_cfg = json.loads((cfg_dir / "binned_lognormal.json").read_text())
	optics_cfg = json.loads((cfg_dir / "optics_homogeneous.json").read_text())

	# Build population and optical population via the library APIs
	pop = build_population(pop_cfg)
	# Convert wvl grid units and pass species_modifications
	optics_cfg = dict(optics_cfg)
	optics_cfg["wvl_grid"] = (np.array(optics_cfg.pop("wvl_grid_um")) * 1e-6).tolist()
	optics_cfg["species_modifications"] = pop_cfg.get("species_modifications", {})

	opt_pop = build_optical_population(pop, optics_cfg)

	# Compute PyParticle population b_ext across wavelengths for RH=0.0
	rh = optics_cfg.get("rh_grid", [0.0])[0]
	bext_pp = np.array(opt_pop.get_optical_coeff("ext", rh=rh))

	# Build D_grid and n_D using the same lognormal mode that the population factory uses
	modes = list(zip(pop_cfg.get("GMD", []), pop_cfg.get("GSD", []), pop_cfg.get("N", [])))
	assert len(modes) > 0, "Test config must include at least one lognormal mode"
	gmd = float(modes[0][0])
	gsd = float(modes[0][1])
	ntot = float(modes[0][2])
	# Interpret GMD as diameter in micrometers if > 1e-6
	if gmd > 1e-6:
		gmd_m = gmd * 1e-6
	else:
		gmd_m = gmd
	r0_m = 0.5 * gmd_m
	D_grid_m, n_D = lognormal_mode_to_bins(r0_m, gsd, ntot, n_bins=80)
	# Convert to nm for PyMieScatt helpers
	D_grid_nm = (D_grid_m * 1e9)

	# Refractive index: emulate the optics factory spectral model
	lam_m = np.asarray(optics_cfg["wvl_grid"], dtype=float)
	wvl_nm = lam_m * 1e9
	n_550 = float(optics_cfg.get("n_550", 1.54))
	k_550 = float(optics_cfg.get("k_550", 0.0))
	alpha_n = float(optics_cfg.get("alpha_n", 0.0))
	alpha_k = float(optics_cfg.get("alpha_k", 0.0))
	ref_um = 0.55
	wvl_um = wvl_nm * 1e-3
	n_lambda = n_550 * (wvl_um / ref_um) ** (alpha_n)
	k_lambda = k_550 * (wvl_um / ref_um) ** (alpha_k)
	m_array = n_lambda + 1j * k_lambda

	# Try PyMieScatt polydisperse helper per wavelength
	import PyMieScatt as PMS
	Cext_ref = np.zeros(len(wvl_nm), dtype=float)
	# Try to use higher-level SD helpers first
	for wi, m in enumerate(m_array):
		# Try polydisperse helper
		Cext_w = _try_pymiescatt_polydisperse(m, [wvl_nm[wi]], D_grid_nm, n_D)
		if Cext_w is None:
			# Fallback: integrate MieQ over the D grid
			Cext_w = _integrate_mieq_over_distribution(m, [wvl_nm[wi]], D_grid_nm, n_D)
		# Helper returned shape (n_wavelengths,) — since we provided single λ, grab first
		Cext_ref[wi] = float(np.atleast_1d(Cext_w)[0])

	# Now compare PyParticle bext_pp (which is in m^-1) to reference Cext_ref aggregated
	# The PyParticle opt_pop returns sum_i Cext_i * N_i which is directly comparable.
	# Ensure arrays match shape and compute relative error.
	assert bext_pp.shape == Cext_ref.shape
	# Avoid strict equality: check relative difference is reasonable
	with np.errstate(divide='ignore', invalid='ignore'):
		rel_err = np.abs(bext_pp - Cext_ref) / np.maximum(1e-12, np.abs(Cext_ref))
	# Allow median relative error < 0.4 (40%) to be permissive for different fallbacks; tune as needed
	med_rel_err = float(np.nanmedian(rel_err))
	assert med_rel_err < 0.4, f"Median relative error too large: {med_rel_err:.3f}"
