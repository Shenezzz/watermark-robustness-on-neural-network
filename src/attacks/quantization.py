"""
Simulated post-training INT8 weight quantization.
 
We use a fake quantization (quantize and immediatly dequantize). The reasin is pytorch Conv expects weights in floatinf point.
Quantization and dequantization reduces the numerical precision and introduces error. Eg: 0.1385 -> 117 -> 0.1373
"""

import copy
 
import torch

def _fake_quantize_tensor(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    levels = 2 ** bits - 1
    w_min, w_max = w.min(), w.max()
    scale = (w_max - w_min) / levels
    if scale == 0:
        return w.clone()  # degenerate all-same-value tensor, nothing to quantize
    q = torch.round((w - w_min) / scale).clamp(0, levels)
    return q * scale + w_min

def apply_int8_weight_quantization(model: torch.nn.Module, bits: int = 8) -> torch.nn.Module:
    """
    Returns a NEW model (deep copy) with every Conv2d/Linear weight tensor
    passed through simulated bits-bit quantization. The original model is
    left untouched.
    """
    attacked = copy.deepcopy(model)
    with torch.no_grad():
        for module in attacked.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                module.weight.data = _fake_quantize_tensor(module.weight.data, bits=bits)
                # Biases are typically left at full precision even in real
                # INT8 quantization schemes (they're a tiny fraction of
                # total params) -- left unquantized here to match that.
    return attacked