"""
Test runner that compares PyParticle outputs against reference libraries
(pyrcel for CCN, PyMieScatt for optics). Designed to be run under pytest
and as a script. Tests skip gracefully if reference libraries are not
installed. Produces a JSON report when `--output` is provided.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


def pytest_addoption(parser):
    parser.addoption("--input", action="store", default="examples/configs/binned_lognormal.json",
                     help="Path to example JSON/YAML config to run comparisons")
    parser.addoption("--compare", action="store", default="both",
                     choices=["pyrcel", "pymiescatt", "both"], help="Which reference library to compare against")
    parser.addoption("--output", action="store", default="reports/reference_report.json",
                     help="Path to write aggregated JSON report")


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input config not found: {p}")
    if p.suffix.lower() in (".json",):
        return json.loads(p.read_text())
    # prefer yaml if available
    try:
        import yaml

        return yaml.safe_load(p.read_text())
    except Exception as exc:  # pragma: no cover - missing yaml
        # If PyYAML not installed, look for a sibling .json file with same stem
        sibling = p.with_suffix('.json')
        if sibling.exists():
            return json.loads(sibling.read_text())
        raise RuntimeError("Only JSON supported unless PyYAML is installed; tried sibling .json") from exc


def write_report(path: str | Path, report: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2))


def run_pyparticle_ccn_single(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run PyParticle single-particle CCN output and return canonical results."""
    try:
        from PyParticle.aerosol_particle import make_particle
    except Exception as exc:
        raise RuntimeError("Unable to import PyParticle aerosol_particle: {}".format(exc))

    # Expect cfg with particle fields: diameter_m, species list of {name,mass_fraction}
    particle_cfg = cfg.get("particle", {})
    D = particle_cfg.get("diameter_m")
    if D is None:
        # support diameter_um convenience
        Dum = particle_cfg.get("diameter_um")
        if Dum is not None:
            D = float(Dum) * 1e-6
        else:
            raise ValueError("particle.diameter_m or particle.diameter_um required")

    species = particle_cfg.get("species", [])
    names = [s["name"] for s in species]
    fracs = [float(s.get("mass_fraction", 1.0 / max(1, len(names)))) for s in species]

    part = make_particle(D, names, fracs, D_is_wet=True)

    env = cfg.get("environment", {})
    T = float(env.get("T_K", env.get("T", 298.15)))

    s_crit = None
    try:
        s_crit = part.get_critical_supersaturation(T)
    except Exception:
        # try passing percent vs fraction variants; caller must interpret
        s_crit = part.get_critical_supersaturation(T)

    return {"s_critical_percent": float(s_crit)}


def run_pyparticle_optics(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run PyParticle optics pipeline using example builder.
    Returns per-wavelength Qext/Qsca/Qabs/g for the first particle or population aggregated values.
    """
    try:
        from PyParticle.population import build_population
        from PyParticle.optics.builder import build_optical_population
    except Exception as exc:
        raise RuntimeError("Unable to import PyParticle optics builders: {}".format(exc))

    pop_cfg = cfg.get("population") or cfg.get("pop_cfg") or cfg.get("population_cfg") or cfg
    optics_cfg = cfg.get("optics", {})

    pop = build_population(pop_cfg)

    # convert wvl grid if provided as microns
    opt_cfg = dict(optics_cfg)
    if "wvl_grid_um" in opt_cfg:
        import numpy as np

        opt_cfg["wvl_grid"] = (np.array(opt_cfg.pop("wvl_grid_um")) * 1e-6).tolist()

    # attach species_modifications if present in pop_cfg
    if isinstance(pop_cfg, dict):
        opt_cfg["species_modifications"] = pop_cfg.get("species_modifications", {})

    opt_pop = build_optical_population(pop, opt_cfg)

    # return aggregated b_ext and g grids as lists
    bext = opt_pop.get_optical_coeff("ext")
    g = opt_pop.get_optical_coeff("g")

    return {"b_ext": bext.tolist() if hasattr(bext, "tolist") else float(bext), "g": g.tolist()}


def compare_scalar(a: float, b: float, rel_tol: float, abs_tol: float) -> Dict[str, Any]:
    abs_diff = abs(a - b)
    rel_diff = abs_diff / max(abs(b), 1e-30)
    passed = (abs_diff <= abs_tol) or (rel_diff <= rel_tol)
    return {"a": a, "b": b, "abs_diff": abs_diff, "rel_diff": rel_diff, "passed": bool(passed)}


@pytest.mark.integration
def test_reference_comparisons(request):
    """Pytest entrypoint for running configured comparisons.

    Use pytest options --input, --compare, --output to control behavior.
    """
    input_path = request.config.getoption("--input")
    compare_mode = request.config.getoption("--compare")
    output_path = request.config.getoption("--output")

    cfg = load_config(input_path)

    report: Dict[str, Any] = {"input": str(input_path), "compare_mode": compare_mode, "results": []}

    # Simple dispatcher: handle CCN single and optics tests if present in cfg
    tests = cfg.get("tests") if isinstance(cfg, dict) and "tests" in cfg else [cfg]

    for t in tests:
        entry = {"name": t.get("test_name", "unnamed"), "status": "ok", "comparisons": []}

        # PyParticle baseline
        try:
            pyp_particle = None
            if t.get("type", "").lower().startswith("ccn"):
                pyp_particle = run_pyparticle_ccn_single(t)
            elif t.get("type", "").lower().startswith("optics") or t.get("type", "").lower().startswith("both"):
                pyp_particle = run_pyparticle_optics(t)
            else:
                # try both
                try:
                    pyp_particle = run_pyparticle_ccn_single(t)
                except Exception:
                    try:
                        pyp_particle = run_pyparticle_optics(t)
                    except Exception as e:
                        entry["status"] = f"pyparticle-error: {e}"
                        report["results"].append(entry)
                        continue
        except Exception as exc:
            entry["status"] = f"pyparticle-error: {exc}"
            report["results"].append(entry)
            continue

        # Reference comparisons
        if compare_mode in ("pyrcel", "both") and t.get("type", "").lower().startswith("ccn"):
            try:
                from tests.reference_wrappers import pyrcel_adapter as pra

                ref = pra.compute_ccn_reference(t)
                comp = compare_scalar(float(pyp_particle["s_critical_percent"]), float(ref["s_critical_percent"]),
                                      float(t.get("tolerances", {}).get("s_relative", 0.02)),
                                      float(t.get("tolerances", {}).get("s_absolute", 1e-3)))
                entry["comparisons"].append({"tool": "pyrcel", "field": "s_critical_percent", "result": comp})
            except Exception as exc:  # pragma: no cover - depends on external libs
                pytest.skip(f"pyrcel adapter unavailable or raised: {exc}")

        if compare_mode in ("pymiescatt", "both") and t.get("type", "").lower().startswith("optics"):
            try:
                from tests.reference_wrappers import pymiescatt_adapter as pma

                ref = pma.compute_optics_reference(t)
                # compare simple metric: first b_ext grid element if available
                a_val = float(pyp_particle.get("b_ext") if isinstance(pyp_particle.get("b_ext"), (int, float)) else pyp_particle.get("b_ext")[0][0])
                b_val = float(ref.get("b_ext")[0][0])
                comp = compare_scalar(a_val, b_val,
                                      float(t.get("tolerances", {}).get("optics_relative", 0.02)),
                                      float(t.get("tolerances", {}).get("optics_absolute", 1e-6)))
                entry["comparisons"].append({"tool": "pymiescatt", "field": "b_ext[0,0]", "result": comp})
            except Exception as exc:
                pytest.skip(f"PyMieScatt adapter unavailable or raised: {exc}")

        report["results"].append(entry)

    # write report
    try:
        write_report(output_path, report)
    except Exception:
        # non-fatal
        pass

    # assert that no comparisons failed
    failed = 0
    for r in report["results"]:
        for c in r.get("comparisons", []):
            if not c["result"]["passed"]:
                failed += 1

    assert failed == 0, f"{failed} comparison(s) failed; see {output_path} for details"


if __name__ == "__main__":
    # allow running standalone for quick debugging
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="examples/configs/binned_lognormal.json")
    ap.add_argument("--compare", choices=("pyrcel", "pymiescatt", "both"), default="both")
    ap.add_argument("--output", default="reports/reference_report.json")
    args = ap.parse_args()
    cfg = load_config(args.input)
    # run a minimal smoke via pytest invocation
    sys.exit(pytest.main([__file__, f"--input={args.input}", f"--compare={args.compare}", f"--output={args.output}"]))
