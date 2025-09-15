import importlib
import pytest

if importlib.util.find_spec("PyMieScatt") is None:
	pytest.skip("PyMieScatt not installed: integration tests skipped", allow_module_level=True)
