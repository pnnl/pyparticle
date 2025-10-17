# src/pyparticle/data.py
from __future__ import annotations
import os
from pathlib import Path
from importlib import resources as ir

_SPEC_ENV = "PYPARTICLE_SPECIES_PATH"    # preferred override for examples
_DATA_ENV = "PYPARTICLE_DATA_PATH"       # legacy: expects a 'species_data' subdir

def _env_species_dir() -> Path | None:
    sp = os.environ.get(_SPEC_ENV)
    if sp:
        p = Path(sp).expanduser()
        if p.is_dir():
            return p
    dp = os.environ.get(_DATA_ENV)
    if dp:
        p = Path(dp).expanduser() / "species_data"
        if p.is_dir():
            return p
    return None

def species_open(name: str, encoding: str = "utf-8"):
    """
    Open a species data file:
      - ${PYPARTICLE_SPECIES_PATH}/{name}
      - ${PYPARTICLE_DATA_PATH}/species_data/{name}
      - packaged: pyparticle/species/data/{name}
    """
    # 1) Env overrides (examples/tests)
    envdir = _env_species_dir()
    if envdir:
        fp = envdir / name
        if fp.is_file():
            return fp.open("r", encoding=encoding)

    # 2) Packaged resource (works for wheels/zips)
    res = ir.files("pyparticle.species").joinpath("data", name)
    with ir.as_file(res) as p:
        return open(p, "r", encoding=encoding)

def species_path(name: str) -> Path:
    """
    Return a filesystem Path to the species data file (materialized if zipped).
    """
    envdir = _env_species_dir()
    if envdir:
        fp = envdir / name
        if fp.is_file():
            return fp
    res = ir.files("pyparticle.species").joinpath("data", name)
    with ir.as_file(res) as p:
        return Path(p)
