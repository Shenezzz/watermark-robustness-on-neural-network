"""e.g. 'layer3.0.conv1' -> model.layer3[0].conv1"""
def get_layer_by_name(model, layer_name):
    module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module