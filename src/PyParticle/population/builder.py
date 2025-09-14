from .factory.registry import discover_population_types
# import inspect

class PopulationBuilder:
    def __init__(self, config):
        self.config = config
    
    def build(self):
        type_name = self.config.get("type")
        if not type_name:
            raise ValueError("Config must include a 'type' key.")
        types = discover_population_types()
        if type_name not in types:
            raise ValueError(f"Unknown population type: {type_name}")
        cls = types[type_name]
        #sig = inspect.signature(cls.__init__)
        # Exclude 'self' and 'type' from kwargs
        #valid_params = [p for p in sig.parameters if p != 'self' and p != 'type']
        #filtered_kwargs = {k: v for k, v in self.config.items() if k in valid_params}
        return cls(self.config)

def build_population(config):
    return PopulationBuilder(config).build()