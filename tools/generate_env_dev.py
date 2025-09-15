"""Generate a developer conda environment YAML from the project's base environment.

This script reads `environment.yml` at the repository root, appends a set of
developer dependencies (integration test dependencies and tooling), and writes
`environment-dev.yml` with name `pyparticle-dev`.

Usage:
    python tools/generate_env_dev.py

The developer dependency set can be extended by editing the DEV_DEPS list
below or by providing a JSON file with additional pip/conda packages.
"""
from __future__ import annotations

try:
    import yaml
except Exception:
    yaml = None
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
ENV_IN = ROOT / "environment.yml"
ENV_OUT = ROOT / "environment-dev.yml"

# Developer extras to append. Keep these conservative and explicit.
DEV_CONDA = [
    # optional conda packages for development / integration
    "pytest",
    "pytest-cov",
    "pyyaml",
]

DEV_PIP = [
    # integration/reference libs
    "pyrcel",
    "PyMieScatt",
    # plotting / notebook helpers
    "jupyter",
]


def load_base_env(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Base environment file not found: {path}")
    txt = path.read_text()
    if yaml is not None:
        return yaml.safe_load(txt)
    # minimal fallback parser: look for lines under dependencies and pip section
    out = {}
    lines = [l.rstrip() for l in txt.splitlines()]
    cur = None
    deps = []
    pip_list = []
    for ln in lines:
        if ln.strip().startswith("name:"):
            out["name"] = ln.split(":", 1)[1].strip()
        if ln.strip().startswith("dependencies:"):
            cur = "deps"
            continue
        if cur == "deps":
            if ln.strip().startswith("- pip:"):
                cur = "pip"
                continue
            if ln.strip().startswith("-"):
                deps.append(ln.strip().lstrip("-").strip())
        elif cur == "pip":
            if ln.strip().startswith("-"):
                pip_list.append(ln.strip().lstrip("-").strip())
    out["dependencies"] = deps + [{"pip": pip_list}]
    return out


def merge_env(base: dict, conda_extra: list[str], pip_extra: list[str]) -> dict:
    out = dict(base)
    out["name"] = "pyparticle-dev"

    deps = out.get("dependencies", [])
    # ensure conda-style deps exist (plain strings)
    existing_conda = [d for d in deps if isinstance(d, str)]
    # append unique extras
    for pkg in conda_extra:
        if pkg not in existing_conda:
            deps.append(pkg)

    # handle pip subsection
    pip_section = None
    for d in deps:
        if isinstance(d, dict) and "pip" in d:
            pip_section = d
            break
    if pip_section is None:
        pip_section = {"pip": []}
        deps.append(pip_section)

    existing_pip = pip_section.get("pip", [])
    for pkg in pip_extra:
        if pkg not in existing_pip:
            existing_pip.append(pkg)
    pip_section["pip"] = existing_pip

    out["dependencies"] = deps
    return out


def write_env(path: Path, data: dict):
    if yaml is not None:
        with path.open("w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
        print(f"Wrote developer environment to: {path}")
        return

    # Minimal YAML writer fallback
    lines = []
    lines.append(f"name: {data.get('name','pyparticle-dev')}")
    lines.append("channels:")
    # preserve channels from base if present
    base_channels = data.get("channels", ["conda-forge", "defaults"])
    for ch in base_channels:
        lines.append(f"  - {ch}")
    lines.append("dependencies:")
    for d in data.get("dependencies", []):
        if isinstance(d, str):
            lines.append(f"  - {d}")
        elif isinstance(d, dict) and "pip" in d:
            lines.append("  - pip:")
            for pkg in d["pip"]:
                lines.append(f"    - {pkg}")
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote developer environment (fallback) to: {path}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", default=str(ENV_IN))
    p.add_argument("--out", dest="outfile", default=str(ENV_OUT))
    args = p.parse_args(argv)

    base = load_base_env(Path(args.infile))
    merged = merge_env(base, DEV_CONDA, DEV_PIP)
    write_env(Path(args.outfile), merged)


if __name__ == "__main__":
    main()
