#!/usr/bin/env python3
"""Inspect population built from example config for debugging.

Prints particle ids, species names, types and masses to help diagnose issues
when calling library functions (like get_critical_supersaturation).
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

cfg_path = ROOT / "examples" / "configs" / "ccn_single_na_cl.json"
if not cfg_path.exists():
    print("Config not found:", cfg_path)
    sys.exit(2)

from pyparticle.population.builder import build_population

cfg = json.loads(cfg_path.read_text())
pop_cfg = cfg.get("population")
if pop_cfg is None:
    print("No population block in config")
    sys.exit(0)

pop = build_population(pop_cfg)
print("Population ids:", pop.ids)
for pid in pop.ids:
    p = pop.get_particle(pid)
    try:
        names = [s.name for s in p.species]
    except Exception:
        names = repr(p.species)
    print(f"--- particle id={pid}")
    print(" species names:", names)
    try:
        masses = list(p.masses)
        print(" masses:", masses)
    except Exception:
        print(" masses: (unreadable)")
    try:
        has_h2o = any(s.name.upper() == 'H2O' for s in p.species)
        print(" has H2O:", has_h2o)
    except Exception:
        print(" has H2O: could not determine")

print("Done")
