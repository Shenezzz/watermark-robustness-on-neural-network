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

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from src.data.loaders import get_cifar_loaders, get_mnist_loaders

from src.training.loop import train_one_epoch



def _get_loader_fn(dataset):
    return {"cifar10": get_cifar_loaders, "mnist": get_mnist_loaders}[dataset]


def _param(m):
    # After the first prune the trainable tensor is weight_orig; `weight` becomes
    # weight_orig * weight_mask. For un-pruned positions the two coincide.
    return m.weight_orig if hasattr(m, "weight_orig") else m.weight

def _prune_by_scores(m, scores, amount):
    """Remove the `amount` fraction of still-alive weights with the lowest score."""
    mask = m.weight_mask.clone() if hasattr(m, "weight_mask") else torch.ones_like(m.weight)
    flat_mask = mask.flatten()
    alive = flat_mask.bool()
    k = int(round(amount * int(alive.sum())))
    if k == 0:
        return
    w = _param(m).detach().flatten()
    s = scores.flatten().clone()
    # Dead units give exactly-zero scores; break those ties by magnitude so the
    # choice is not an arbitrary index-order artefact.
    s = torch.where(s == 0, w.abs() * 1e-30, s)
    s[~alive] = float("inf")  # never re-select already-pruned weights
    flat_mask[torch.topk(s, k, largest=False).indices] = 0
    prune.custom_from_mask(m, "weight", flat_mask.view_as(mask))


def _sensitivity_scores(model, modules, loader, device, n_batches):
    model.eval()  # leave BatchNorm running statistics untouched while scoring
    params = [_param(m) for m in modules]
    acc_sqaured_gradient = [torch.zeros_like(p) for p in params]
    seen = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        model.zero_grad()
        F.cross_entropy(model(images), labels).backward()
        for a, p in zip(acc_sqaured_gradient, params):
            a += p.grad.detach() ** 2
        seen += 1
        if seen >= n_batches:
            break
    model.zero_grad()
    return [(p.detach() ** 2) * (a / seen) for p, a in zip(params, acc_sqaured_gradient)]

def apply_pruning(model, cfg: dict):
    # sparsity = cfg["sparsity"]  # Eg: 0.5 = zero out the smallest 50% of each layer's weights
    # attacked = copy.deepcopy(model)
    # for module in attacked.modules():
    #     if isinstance(module, (nn.Conv2d, nn.Linear)):
    #         prune.l1_unstructured(module, name="weight", amount=sparsity) # L1 unstructured pruning = zero out individual weights having the smallest absolute magnitude.
    #         prune.remove(module, "weight")  # Element wise mulitplication of Mask and weight
    # return attacked
    
    sparsity = cfg["sparsity"]
    rounds = cfg.get("rounds", 1)
    ft_epochs = cfg.get("finetune_epochs", 0)
    criterion = cfg.get("criterion", "magnitude")
    if criterion not in ("magnitude", "sensitivity"):
        raise ValueError(f"Unknown criterion {criterion!r} (expected 'magnitude' or 'sensitivity')")
    
    attacked = copy.deepcopy(model)
    modules = [m for m in attacked.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    per_round = 1 - (1 - sparsity) ** (1 / rounds) # Equation to get the deseired sparsity per round. if sparsity is 0.5, per round there would "0.12944943670387588" weight prunning.
    device = next(attacked.parameters()).device
    
    train_loader = None
    
    if ft_epochs > 0 or criterion == "sensitivity":
        dataset = cfg.get("_dataset", cfg.get("dataset", "cifar10"))
        train_loader, _ = _get_loader_fn(dataset)(batch_size=cfg.get("batch_size", 128))
 
    for r in range(1, rounds + 1):
        if criterion == "magnitude":
            for m in modules:
                prune.l1_unstructured(m, name="weight", amount=per_round)
        else:
            scores = _sensitivity_scores(attacked, modules, train_loader, device,
                                            cfg.get("sensitivity_batches", 50))
            for m, s in zip(modules, scores):
                _prune_by_scores(m, s, per_round)
        
        if ft_epochs > 0:
            trainable = [p for p in attacked.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(trainable, lr=cfg.get("lr", 0.01), momentum=0.9, weight_decay=5e-4)
            for e in range(ft_epochs):
                loss, acc = train_one_epoch(attacked, train_loader, optimizer, device)
                print(f"  [pruning/{criterion}] round {r}/{rounds} epoch {e + 1}/{ft_epochs} | "
                      f"train_loss {loss:.4f} | train_acc {acc:.4f}")
                
    for m in modules:
        prune.remove(m, "weight")  # bake the mask in permanently
    return attacked
                
            