"""Check conda env inheritance between pyparticle-partmc and pyparticle-dev.

This script compares the list of top-level packages installed in the two
environments. It will exit with code 0 if pyparticle-dev appears to contain
all packages found in pyparticle-partmc (names only). It will raise SystemExit
with a non-zero code otherwise.

Run from project root (requires conda available in PATH):

    python tools/check_env_inheritance.py

"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Set


def conda_list(env_name: str) -> Set[str]:
    cmd = ["conda", "list", "-n", env_name, "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"Failed to run conda list for {env_name}: {p.stderr.strip()}")
    data = json.loads(p.stdout)
    return {pkg["name"] for pkg in data}


def main():
    base = "pyparticle-partmc"
    dev = "pyparticle-dev"
    print(f"Checking conda envs: {base} -> {dev}")
    base_pkgs = conda_list(base)
    dev_pkgs = conda_list(dev)

    missing = base_pkgs - dev_pkgs
    if missing:
        print("The following packages are in pyparticle-partmc but missing from pyparticle-dev:")
        for m in sorted(missing):
            print(" -", m)
        raise SystemExit(2)

    print("pyparticle-dev appears to include all packages from pyparticle-partmc (by name).")


if __name__ == "__main__":
    main()
