"""e.g. 'layer3.0.conv1' -> model.layer3[0].conv1"""
def get_layer_by_name(model, layer_name):
    module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module

# In-place layer replacement helper for DeepEclipse.
def set_layer_by_name(model, layer_name, new_module):
    """e.g. 'layer3.0.conv1' -> model.layer3[0].conv1 = new_module"""
    parts = layer_name.split(".")
    module = model
    for part in parts[:-1]:
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    last = parts[-1]
    if last.isdigit():
        module[int(last)] = new_module
    else:
        setattr(module, last, new_module)