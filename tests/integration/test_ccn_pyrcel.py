import importlib
import pytest

if importlib.util.find_spec("pyrcel") is None:
	pytest.skip("pyrcel not installed: integration tests skipped", allow_module_level=True)
