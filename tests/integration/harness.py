from __future__ import annotations
from typing import Dict, Any
import importlib.util, pathlib

# Load runner modules by path to avoid package import issues during pytest collection
RDIR = pathlib.Path(__file__).parent / "runners"

def _load_runner(name: str):
    path = RDIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_base_mod = _load_runner("base_runner")
_optics_mod = _load_runner("optics_runner")

run_base = _base_mod.run_base
run_optics = _optics_mod.run_optics

def run_scenario_file(cfg: Dict[str, Any], base_dir, tmp_dir):
    """
    Routes scenario dicts to the right runner.
    cfg must include: module: 'base' | 'optics'
    When module='optics', cfg must also include an optics config with 'type' (morphology).
    """
    module = (cfg.get("module") or "").strip().lower()
    if module == "base":
        return run_base(cfg, base_dir=base_dir, tmp_dir=tmp_dir)
    if module == "optics":
        return run_optics(cfg, base_dir=base_dir, tmp_dir=tmp_dir)
    raise AssertionError("Scenario must set module: 'base' or 'optics'")
