"""
Magnitude-based (L1) unstructured pruning.

For each Conv2d/Linear layer independently, zeroes out the `sparsity`
fraction of weights with the smallest absolute value. This is the standard
first pruning attack in the watermarking-robustness literature -- global
pruning (across all layers at once, by one shared threshold) is a common
variant too, but per-layer is simpler to reason about and is what most
papers report as the baseline.

Uses torch.nn.utils.prune rather than a hand-rolled implementation --
well-tested, standard PyTorch API. prune.remove() bakes the mask
permanently into .weight (removing the weight_orig/weight_mask
reparametrization) so every downstream piece of code that reads
module.weight.data directly keeps working unchanged.
"""

import copy

import torch.nn as nn
import torch.nn.utils.prune as prune


def apply_pruning(model, cfg: dict):
    sparsity = cfg["sparsity"]  # Eg: 0.5 = zero out the smallest 50% of each layer's weights
    attacked = copy.deepcopy(model)
    for module in attacked.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=sparsity) # L1 unstructured pruning = zero out individual weights having the smallest absolute magnitude.
            prune.remove(module, "weight")  # Element wise mulitplication of Mask and weight
    return attacked