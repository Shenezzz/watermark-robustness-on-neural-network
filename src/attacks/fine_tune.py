"""
Fine-tuning attacks: FTAL (Fine-Tune All Layers) and RTAL (Retrain All
Layers... except this only retrains the last layer -- the naming in the
literature is inconsistent; what matters is the distinction: FTAL updates
every parameter, RTAL freezes everything except the final classifier layer).

Unlike the other attacks, this one needs real training data and multiple
epochs -- it reuses train_one_epoch from src/training/loop.py directly,
the same function baseline training and watermark embedding both use.

RTAL is the interesting robustness test for a WEIGHT-based watermark like
Uchida's: if the watermarked layer is anywhere except the final classifier,
RTAL should leave it completely untouched (gradients never reach it), so
watermark_retention should come back EXACTLY 1.0 -- not approximately, but
bit-for-bit identical to before the attack. That's a useful sanity check in
itself, separate from being an interesting attack result.
"""

import copy

import torch

from src.data.loaders import get_cifar_loaders
from src.training.loop import train_one_epoch


def apply_fine_tune(model: torch.nn.Module, cfg: dict) -> torch.nn.Module:
    mode = cfg["mode"]  # "ftal" or "rtal"
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    batch_size = cfg.get("batch_size", 128)

    device = next(model.parameters()).device
    attacked = copy.deepcopy(model)

    if mode == "ftal":
        for param in attacked.parameters():
            param.requires_grad = True
    elif mode == "rtal":
        for name, param in attacked.named_parameters():
            param.requires_grad = name.startswith("fc.")  # only the final classifier layer
    else:
        raise ValueError(f"Unknown fine-tune mode: {mode!r} (expected 'ftal' or 'rtal')")

    trainable_params = [p for p in attacked.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=5e-4)

    train_loader, _ = get_cifar_loaders(batch_size=batch_size)
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(attacked, train_loader, optimizer, device)
        print(f"  [fine_tune/{mode}] epoch {epoch + 1}/{epochs} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f}")

    return attacked