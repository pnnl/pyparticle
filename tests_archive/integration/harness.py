from __future__ import annotations
from typing import Dict, Any
from .runners.base_runner import run_base
from .runners.optics_runner import run_optics

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
