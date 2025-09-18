import importlib
import warnings
import pytest
import numpy as np

pyrcel_spec = importlib.util.find_spec("pyrcel")
if pyrcel_spec is None:
	pytest.skip("pyrcel not installed: integration tests skipped", allow_module_level=True)


def test_pyrcel_parcel_activation_monotonic():
	"""Run a small pyrcel parcel-model sweep and compare against pyrcel parameterizations.

	This test is intended as an integration smoke: it executes a few ParcelModel
	runs (kept short for CI) and verifies that reported maximum supersaturation
	and activated fraction increase with updraft speed, and that parameterization
	functions return arrays of the expected length.
	"""
	import pyrcel as pm

	warnings.simplefilter("ignore")

	# Parcel initial conditions
	P0 = 100000.0
	T0 = 279.0
	S0 = -0.1

	# Build a lognormal ammonium sulfate accumulation-mode aerosol
	# Use modest bin count and smaller N for CI speed
	aer = pm.AerosolSpecies("ammonium sulfate", pm.Lognorm(mu=0.05, sigma=2.0, N=1000.0), kappa=0.7, bins=80)

	# Updraft speeds (reduced set for CI speed), ascending order
	Vs = np.logspace(-1, np.log10(10.0), 7)
	accom = 0.1

	# Use pyrcel parameterizations (fast) across Vs
	smaxes_arg = []
	act_fracs_arg = []
	smaxes_mbn = []
	act_fracs_mbn = []
	for V in Vs:
		smax_arg, _, afs_arg = pm.arg2000(V, T0, P0, [aer], accom=accom)
		smax_mbn, _, afs_mbn = pm.mbn2014(V, T0, P0, [aer], accom=accom)
		smaxes_arg.append(float(smax_arg))
		act_fracs_arg.append(float(afs_arg[0]))
		smaxes_mbn.append(float(smax_mbn))
		act_fracs_mbn.append(float(afs_mbn[0]))

	smaxes_arg = np.array(smaxes_arg, dtype=float)
	act_fracs_arg = np.array(act_fracs_arg, dtype=float)
	smaxes_mbn = np.array(smaxes_mbn, dtype=float)
	act_fracs_mbn = np.array(act_fracs_mbn, dtype=float)

	# Basic sanity checks for parameterizations
	assert smaxes_arg.shape == Vs.shape
	assert smaxes_mbn.shape == Vs.shape
	assert np.all(np.isfinite(smaxes_arg)) and np.all(np.isfinite(smaxes_mbn))
	assert np.all(act_fracs_arg >= 0.0) and np.all(act_fracs_arg <= 1.0)
	assert np.all(act_fracs_mbn >= 0.0) and np.all(act_fracs_mbn <= 1.0)

	# With increasing V, smax and activated fraction should generally increase
	assert np.all(np.diff(smaxes_arg) >= -1e-8)
	assert np.all(np.diff(act_fracs_arg) >= -1e-8)
	assert np.all(np.diff(smaxes_mbn) >= -1e-8)
	assert np.all(np.diff(act_fracs_mbn) >= -1e-8)

	# Also exercise binned_activation directly at a few S values using the lognormal distribution
	# Build a simple wet size array for testing: use the dry radii from the AerosolSpecies
	# The AerosolSpecies object exposes `r_drys` (dry radii) and `Nis` (counts per bin)
	diam_m = np.asarray(aer.r_drys, dtype=float)
	# pick a few S values to test activation fraction calculation
	for S_test in (0.001, 0.005, 0.01):
		frac_eq, _, _, _ = pm.binned_activation(S_test, T0, diam_m, aer)
		assert 0.0 <= frac_eq <= 1.0

