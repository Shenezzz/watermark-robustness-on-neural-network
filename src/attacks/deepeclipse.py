"""
DeepEclipse attack, wired into the src/attacks/ registry pattern.

The core obfuscation primitives (base/advanced, linear/conv) are the same
ones verified earlier in deepeclipse_attack.py -- see that file's docstring
for the full mechanism explanation and the numerical function-preservation
checks. This file adds:
  - a thin apply_deepeclipse(model, cfg) wrapper matching the shape every
    other attack in ATTACK_REGISTRY uses
  - in-place layer replacement (setattr on the parent module), since the
    standalone primitives return new layers but don't splice them back
    into an arbitrary model themselves
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.model_utils import get_layer_by_name, set_layer_by_name


# ---------------------------------------------------------------------------
# Detection (Section 4.2 of the paper) -- unchanged from deepeclipse_attack.py
# ---------------------------------------------------------------------------
def frequency_volatility_score(weight: torch.Tensor, window: int = 50) -> float:
    flat = weight.detach().flatten().float()
    freq = torch.fft.fft(flat).real
    if freq.numel() < window + 1:
        window = max(2, freq.numel() // 2)
    kernel = torch.ones(window) / window
    sma = F.conv1d(freq.view(1, 1, -1), kernel.view(1, 1, -1)).flatten()
    return sma.diff().abs().mean().item()


def rank_candidate_layers(model: nn.Module, window: int = 50):
    scored = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            scored.append((name, module, frequency_volatility_score(module.weight, window=window)))
    return sorted(scored, key=lambda t: t[2])


def base_conv_obfuscation(conv: nn.Conv2d):
    device, dtype = conv.weight.device, conv.weight.dtype
    new_weight = F.pad(conv.weight.data, (1, 1, 1, 1), value=0.0)
    old_pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=(conv.kernel_size[0] + 2, conv.kernel_size[1] + 2),
        stride=conv.stride, 
        padding=(old_pad[0] + 1, old_pad[1] + 1),
        dilation=conv.dilation, groups=conv.groups, bias=conv.bias is not None,
    ).to(device, dtype)
    new_conv.weight.data = new_weight
    if conv.bias is not None:
        new_conv.bias.data = conv.bias.data.clone()
    return new_conv


def advanced_conv_obfuscation(conv: nn.Conv2d, next_conv: nn.Conv2d,
                               beta: float = 10.0, mu: float = 0.33, sigma: float = 0.1,
                               stat: str = "min", seed: int = None):
    device, dtype = conv.weight.device, conv.weight.dtype
    generator = torch.Generator(device=device) if seed is not None else None
    if generator is not None:
        generator.manual_seed(seed)

    padded = F.pad(conv.weight.data, (1, 1, 1, 1), value=0.0)
    P = padded.shape[0]
    for p in range(P):
        filt = conv.weight.data[p]
        # In Algorithm 3 (AdvConvObfuscation) of the DeepEclipse paper (USENIX 2024),
        # eps is computed using min(|K^p|). Using median on high-channel layers injects
        # excessive cumulative border noise, causing model collapse.
        scale = filt.abs().min().item() if stat == "min" else filt.abs().median().item()
        noise = torch.normal(mean=mu, std=sigma, size=(1,), generator=generator).item()
        eps_p = beta * scale * noise
        border_mask = torch.ones_like(padded[p])
        border_mask[:, 1:-1, 1:-1] = 0
        padded[p] = padded[p] + border_mask * eps_p

    if generator is not None:
        lam = float(torch.empty(1).uniform_(0.5, 1.5, generator=generator))
    else:
        lam = float(torch.empty(1).uniform_(0.5, 1.5))
    old_pad = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        kernel_size=(conv.kernel_size[0] + 2, conv.kernel_size[1] + 2),
        stride=conv.stride, 
        padding=(old_pad[0] + 1, old_pad[1] + 1),
        dilation=conv.dilation, 
        groups=conv.groups, 
        bias=True,
    ).to(device, dtype)
    new_conv.weight.data = padded * lam
    new_conv.bias.data = (conv.bias.data.clone() * lam) if conv.bias is not None else torch.zeros(P, device=device, dtype=dtype)

    new_next = nn.Conv2d(
        next_conv.in_channels, next_conv.out_channels,
        kernel_size=next_conv.kernel_size, stride=next_conv.stride,
        padding=next_conv.padding, 
        dilation=next_conv.dilation,
        groups=next_conv.groups, 
        bias=next_conv.bias is not None,
    ).to(device, dtype)
    new_next.weight.data = next_conv.weight.data.clone() / lam
    if next_conv.bias is not None:
        new_next.bias.data = next_conv.bias.data.clone()

    return new_conv, new_next

def apply_deepeclipse(model: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    """
    cfg expects:
        mode: "base" or "advanced"
        next_layer: required only for mode == "advanced" -- the conv layer
            immediately downstream of the target (e.g. the other conv in
            the same BasicBlock)
        _watermark_layer: injected automatically by run_attack.py from the
            checkpoint -- the layer being obfuscated
    """
    attacked = copy.deepcopy(model)
    target_name = cfg["_watermark_layer"]
    target_layer = get_layer_by_name(attacked, target_name)
    assert isinstance(target_layer, nn.Conv2d), \
        f"DeepEclipse conv obfuscation requires a Conv2d layer, got {type(target_layer)}"

    mode = cfg.get("mode", "base")
    if mode == "base":
        new_layer = base_conv_obfuscation(target_layer)
        set_layer_by_name(attacked, target_name, new_layer)
    elif mode == "advanced":
        next_name = cfg["next_layer"]
        next_layer = get_layer_by_name(attacked, next_name)
        assert isinstance(next_layer, nn.Conv2d), \
            f"'next_layer' must be a Conv2d layer, got {type(next_layer)}"
        beta = cfg.get("beta", 10.0)
        mu = cfg.get("mu", 0.33)
        sigma = cfg.get("sigma", 0.1)
        stat = cfg.get("stat", "min")
        seed = cfg.get("seed", None)
        new_target, new_next = advanced_conv_obfuscation(
            target_layer, next_layer,
            beta=beta, mu=mu, sigma=sigma, stat=stat, seed=seed,
        )
        set_layer_by_name(attacked, target_name, new_target)
        set_layer_by_name(attacked, next_name, new_next)
    else:
        raise ValueError(f"Unknown DeepEclipse mode: {mode!r} (expected 'base' or 'advanced')")

    return attacked