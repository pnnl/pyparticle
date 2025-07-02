from .factory import create_population

def build_population(type_name, settings, **kwargs):
    """High-level builder function for populations."""
    return create_population(type_name, settings, **kwargs)