import importlib
import pkgutil
import os

def discover_population_types():
    """Discover all population type modules in the types/ submodule."""
    types_pkg = __package__ + ".types"
    types_path = os.path.join(os.path.dirname(__file__), "types")
    population_types = {}
    for _, module_name, _ in pkgutil.iter_modules([types_path]):
        module = importlib.import_module(f"{types_pkg}.{module_name}")
        if hasattr(module, "build"):
            population_types[module_name] = module.build
    return population_types

def create_population(type_name, settings, **kwargs):
    types = discover_population_types()
    if type_name not in types:
        raise ValueError(f"Unknown population type: {type_name}")
    return types[type_name](settings, **kwargs)