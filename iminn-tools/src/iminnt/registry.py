_class_registry = {}

def register_class(name):
    """Decorator to register classes by name"""
    def decorator(cls):
        _class_registry[name] = cls
        return cls
    return decorator

def get_registered_classes():
    """Get all registered classes"""
    return _class_registry.copy()

def get_class_by_name(name):
    """Get a specific class by name"""
    return _class_registry.get(name)

def get_class_names():
    """Get all registered class names for CLI options"""
    return list(_class_registry.keys())