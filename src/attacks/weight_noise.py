"""
Gaussian weight-noise injection.
 
Adds noise scaled relative to EACH layer's own weight std (not a single
global noise scale) -- a layer with naturally larger weights gets
proportionally larger noise, which is the standard way this baseline attack
is implemented in the robustness literature (an absolute noise scale would
be dominated by whichever layer happens to have the largest weights).
"""

import copy
 
import torch

def apply_weight_noise(model: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    relative_std = cfg["relative_std"]  # e.g. 0.05 = noise with std = 5% of each layer's own weight std
    #To make random noise reproducilble allowing us experiment under same condition
    seed = cfg.get("seed", 0)
    generator = torch.Generator().manual_seed(seed)
 
    attacked = copy.deepcopy(model)
    with torch.no_grad():
        for module in attacked.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layer_std = module.weight.data.std().item() # stnadard deviation of the weight of the layer
                # select a random noise which is distributed using Gaussian(0,1) and mulitply it with 
                # the layers standard deviation and relative standard deviation. 
                noise = torch.randn(module.weight.shape, generator=generator) * (layer_std * relative_std) 
                module.weight.data += noise.to(module.weight.device, module.weight.dtype)
    return attacked