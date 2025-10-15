import pathlib, yaml, pytest
from .harness import run_scenario_file

SCEN_DIR = pathlib.Path(__file__).parent / "scenarios"
FILES = sorted(SCEN_DIR.glob("*.yaml"))


def _coerce_number(x):
    if isinstance(x, str):
        s = x.strip()
        try:
            # floats first to catch scientific notation
            if any(c in s for c in ".eE"):
                return float(s)
            # plain ints
            if s.lstrip("+-").isdigit():
                return int(s)
        except Exception:
            return x
    return x


def _sanitize(obj):
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return _coerce_number(obj)


@pytest.mark.integration
@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_scenario(path, tmp_path):
    cfg = _sanitize(yaml.safe_load(path.read_text()))
    run_scenario_file(cfg, base_dir=path.parent, tmp_dir=tmp_path)
