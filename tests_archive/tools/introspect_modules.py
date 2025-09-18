import importlib
import inspect
import json
import pkgutil
import sys
from typing import Dict, List


def detect_package_name() -> str | None:
    for name in ("PyParticle",):
        try:
            importlib.import_module(name)
            return name
        except Exception:
            continue
    # Try adding src to sys.path
    sys.path.insert(0, "src")
    try:
        importlib.import_module("PyParticle")
        return "PyParticle"
    except Exception:
        return None


def enumerate_submodules(pkg_name: str) -> List[str]:
    pkg = importlib.import_module(pkg_name)
    if not hasattr(pkg, "__path__"):
        return []
    names: List[str] = []
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        name = mod.name
        # Exclude private and tests
        if any(part.startswith("_") for part in name.split(".")):
            continue
        if ".tests" in name:
            continue
        names.append(name)
    return names


def public_api(mod_name: str) -> List[str]:
    m = importlib.import_module(mod_name)
    pub = []
    if hasattr(m, "__all__"):
        pub = [n for n in getattr(m, "__all__") if not n.startswith("_")]
    else:
        for k, v in vars(m).items():
            if k.startswith("_"):
                continue
            if inspect.isfunction(v) or inspect.isclass(v):
                pub.append(k)
    return sorted(set(pub))


def main() -> None:
    pkg = detect_package_name()
    if not pkg:
        print(json.dumps({"error": "package_not_found"}))
        raise SystemExit(1)
    subs = enumerate_submodules(pkg)
    mapping: Dict[str, List[str]] = {s: public_api(s) for s in subs}
    out = {"package": pkg, "submodules": subs, "public": mapping}
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
