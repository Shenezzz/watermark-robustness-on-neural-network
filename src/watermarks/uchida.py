"""
Uchida et al., "Embedding Watermarks into Deep Neural Networks" (2017).

Mechanism: pick one conv layer. Average its weights across the output-channel
dimension to get a single representative vector w_bar. Project that vector
through a FIXED secret random matrix X (the key -- generated once, never
trained, saved alongside the watermarked checkpoint). The projection, after
a sigmoid, should equal a secret target bit-string b (the payload).

During fine-tuning, add BCE(sigmoid(X @ w_bar), b) to the ordinary
classification loss. This nudges the layer toward a point in weight-space
that both classifies well AND encodes the bits -- it does not change what
the network computes on real inputs, since there are many equally-good
weight configurations and this loss just steers training toward one that
also satisfies the watermark constraint.

Verification later (including post-attack) only needs: the watermarked
model's current weights, the same layer name, and the same (X, b) key --
all three get saved together in the watermarked checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UchidaWatermark:
    def __init__(self, layer: nn.Conv2d, message_len: int = 32, seed: int = 0):
        self.layer = layer
        self.message_len = message_len

        vector_len = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
        device, dtype = layer.weight.device, layer.weight.dtype

        # Secret key: fixed random projection matrix, generated once and
        # never updated by the optimizer (it's not an nn.Parameter).
        gen = torch.Generator().manual_seed(seed)
        self.X = torch.randn(message_len, vector_len, generator=gen).to(device, dtype)

        # Secret payload: the bit-string you're actually embedding.
        payload_gen = torch.Generator().manual_seed(seed + 1)  # different seed from X
        self.target_bits = (torch.rand(message_len, generator=payload_gen) > 0.5).float().to(device, dtype)

    def _weight_vector(self):
        # Mean across the output-channel dim -> (in_channels, kh, kw) -> flatten
        return self.layer.weight.mean(dim=0).flatten()

    def regularization_loss(self, model=None):
        # Signature matches the extra_loss_fn(model) hook in src/training/loop.py.
        # `model` is unused here since self.layer is already a direct reference,
        # but kept in the signature so this plugs into train_one_epoch as-is.
        w_bar = self._weight_vector()
        logits = self.X @ w_bar  # (message_len,)
        return F.binary_cross_entropy_with_logits(logits, self.target_bits)

    @torch.no_grad()
    def extract_message(self):
        w_bar = self._weight_vector()
        if w_bar.numel() != self.X.shape[1]:
            # The attack changed this layer's weight shape (e.g. DeepEclipse's
            # kernel padding). Extraction is not just noisy here, it's
            # structurally impossible with the original key -- report as
            # fully broken (chance-level) rather than crashing the whole
            # attack sweep on a shape-mismatch error.
            return torch.full_like(self.target_bits, 0.5)
        logits = self.X @ w_bar
        return (torch.sigmoid(logits) > 0.5).float()

    @torch.no_grad()
    def bit_accuracy(self):
        """Watermark retention metric: fraction of bits that still match the
        secret payload. 1.0 = fully intact. ~0.5 = watermark destroyed
        (indistinguishable from random guessing on a binary string)."""
        extracted = self.extract_message()
        return (extracted == self.target_bits).float().mean().item()

    def state_dict(self):
        """Everything needed to re-verify this watermark later, including
        after an attack has modified self.layer's weights."""
        return {
            "message_len": self.message_len,
            "X": self.X.cpu(),
            "target_bits": self.target_bits.cpu(),
        }

    @classmethod
    def from_state_dict(cls, layer: nn.Conv2d, state):
        wm = cls.__new__(cls)
        wm.layer = layer
        wm.message_len = state["message_len"]
        wm.X = state["X"].to(layer.weight.device, layer.weight.dtype)
        wm.target_bits = state["target_bits"].to(layer.weight.device, layer.weight.dtype)
        return wm