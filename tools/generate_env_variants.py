"""Generate environment-dev.yml and environment-partmc.yml from environment.yml

Usage:
  python tools/generate_env_variants.py --write

This script reads `environment.yml` in the repository root, creates two
variants that inherit the base dependencies and append developer and
partmc-specific packages, and writes them to disk.

It prefers to use PyYAML; if not present, it will instruct the user to
install it.
"""
from __future__ import annotations
import sys
from pathlib import Path
from copy import deepcopy

try:
    import yaml
except Exception:  # pragma: no cover - diagnostic path
    yaml = None

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_ENV = REPO_ROOT / "environment.yml"
DEV_ENV = REPO_ROOT / "environment-dev.yml"
PARTMC_ENV = REPO_ROOT / "environment-partmc.yml"

# Extra packages to add
DEV_CONDA = []
DEV_PIP = [
    "pytest",
    "pytest-cov",
    "codecov",
    "pymiescatt",
    "pyrcel",
    "PyMieScatt",
    "jupyter",
]

PARTMC_CONDA = [
    # netCDF4 is normally installed from conda-forge as 'netcdf4'
    "netcdf4",
]
PARTMC_PIP = []


def load_env(path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to run this script. Install with `pip install pyyaml`.")
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)


def dump_env(data: dict, path: Path) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required to run this script. Install with `pip install pyyaml`.")
    with open(path, "w", encoding="utf8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_pip_block(env: dict) -> list:
    """Return the pip list within env['dependencies'], creating it if needed."""
    deps = env.setdefault("dependencies", [])
    # find existing pip block (a dict inside the list)
    for item in deps:
        if isinstance(item, dict) and "pip" in item:
            return item["pip"]
    # not found: append a pip dict
    pip_list = []
    deps.append({"pip": pip_list})
    return pip_list


def add_conda_pkgs(env: dict, pkgs: list[str]) -> None:
    deps = env.setdefault("dependencies", [])
    for pkg in pkgs:
        if pkg not in deps:
            deps.insert(0, pkg)


def make_dev_env(base: dict) -> dict:
    env = deepcopy(base)
    add_conda_pkgs(env, DEV_CONDA)
    pip_list = ensure_pip_block(env)
    for pkg in DEV_PIP:
        if pkg not in pip_list:
            pip_list.append(pkg)
    env["name"] = env.get("name", "pyparticle") + "-dev"
    return env


def make_partmc_env(base: dict) -> dict:
    env = deepcopy(base)
    add_conda_pkgs(env, PARTMC_CONDA)
    pip_list = ensure_pip_block(env)
    for pkg in PARTMC_PIP:
        if pkg not in pip_list:
            pip_list.append(pkg)
    env["name"] = env.get("name", "pyparticle") + "-partmc"
    return env


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    write = "--write" in argv or "-w" in argv
    if not BASE_ENV.exists():
        print(f"Base environment file not found: {BASE_ENV}")
        return 2
    if yaml is None:
        print("PyYAML is required. Install it into your Python (pip install pyyaml) and rerun.")
        return 3

    base = load_env(BASE_ENV)
    dev = make_dev_env(base)
    partmc = make_partmc_env(base)

    print("Generated environment variants in memory:")
    print(f" - dev: {DEV_ENV.name}")
    print(f" - partmc: {PARTMC_ENV.name}")

    if write:
        dump_env(dev, DEV_ENV)
        dump_env(partmc, PARTMC_ENV)
        print(f"Wrote {DEV_ENV} and {PARTMC_ENV}")
    else:
        print("Run with --write to write files to disk.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
