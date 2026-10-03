"""
Run an attack against a watermarked checkpoint and measure the result.

This is the script you'll run once per (model, watermark, attack)
combination. swap --config to point at a different attack config to test
a different attack against the same watermarked model.

Usage:

ResNet-18 :
    python scripts/run_attack.py --config configs/attacks/ResNet18/quantization.yaml
    python scripts/run_attack.py --config configs/attacks/ResNet18/weight_noise.yaml
    python scripts/run_attack.py --config configs/attacks/ResNet18/deepeclipse_base.yaml
    python scripts/run_attack.py --config configs/attacks/ResNet18/deepeclipse_advanced.yaml
    python scripts/run_attack.py --config configs/attacks/ResNet18/pruning.yaml
    python scripts/run_attack.py --config configs/attacks/ResNet18/fine_tune.yaml

    python scripts/run_attack.py --config configs/attacks/ResNet18/quantization_weight_noise.yaml

LeNet5 :
    python scripts/run_attack.py --config configs/attacks/LeNet5/quantization.yaml
    python scripts/run_attack.py --config configs/attacks/LeNet5/weight_noise.yaml
    python scripts/run_attack.py --config configs/attacks/LeNet5/deepeclipse_base.yaml
    python scripts/run_attack.py --config configs/attacks/LeNet5/pruning.yaml
    python scripts/run_attack.py --config configs/attacks/LeNet5/fine_tune.yaml
"""

import argparse  # Used to handle command line arguements
import os  # Provides function that helps to interact with the OS. Commonly used to handle file directories
import sys

import torch  # Commonly used library for constructing neural networks
import yaml  # Used to read configs from yaml files

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # To find the porjects root directory

from src.models.resnet18 import resnet18_cifar
from src.models.lenet5 import lenet5_mnist

from src.data.loaders import get_cifar_loaders, get_mnist_loaders

from src.watermarks.uchida import UchidaWatermark

from src.training.loop import evaluate

from src.utils.model_utils import get_layer_by_name
from src.utils.load_config_and_checkpoint import load_config_and_checkpoint

from src.eval.metrics import task_accuracy_retention, watermark_retention

from src.attacks.quantization import apply_int8_weight_quantization
from src.attacks.weight_noise import apply_weight_noise
from src.attacks.deepeclipse import apply_deepeclipse
from src.attacks.pruning import apply_pruning
from src.attacks.fine_tune import apply_fine_tune

# Creaating a registry of attacks. YAML files inside configs/attacks/ResNet18 contains the arguments that
# needs to be passed to run an attack including the attack name. so when we get an attack name the
# the registry knows which function to call.
ATTACK_REGISTRY = {
    "quantization": lambda model, cfg: apply_int8_weight_quantization(
        model, bits=cfg["bits"]
    ),
    "weight_noise": lambda model, cfg: apply_weight_noise(model, cfg),
    "deepeclipse": lambda model, cfg: apply_deepeclipse(model, cfg),
    "pruning": lambda model, cfg: apply_pruning(model, cfg),
    "fine_tune": lambda model, cfg: apply_fine_tune(model, cfg),
}

MODEL_REGISTRY = {"resnet18_cifar": resnet18_cifar, "lenet5_mnist": lenet5_mnist}
DATA_REGISTRY = {"cifar10": get_cifar_loaders, "mnist": get_mnist_loaders}


def main():
    cfg, device, ckpt, baseline_test_acc = load_config_and_checkpoint(
        "baseline_test_acc"
    )

    model = MODEL_REGISTRY[ckpt.get("model")](num_classes=10).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    # Acces the layer that is watermarked
    layer = get_layer_by_name(model, ckpt["watermark_layer"])
    wm = UchidaWatermark.from_state_dict(layer, ckpt["watermark_state"])

    pre_attack_bit_acc = wm.bit_accuracy()
    print(
        f"Loaded watermarked model -- pre-attack watermark_bit_acc: {pre_attack_bit_acc:.3f} "
        f"(expect 1.000 -- if not, something is wrong before the attack even runs)"
    )

    # Apply attack
    cfg["_watermark_layer"] = ckpt["watermark_layer"]
    attack_fn = ATTACK_REGISTRY[cfg["attack"]]
    attacked_model = attack_fn(model, cfg).to(device)

    # Get a fresh layer reference from the ATTACKED model (some
    # attacks, like this quantization one, deepcopy the model, the old
    # `layer` reference now points at the pre-attack model's weights).
    attacked_layer = get_layer_by_name(attacked_model, ckpt["watermark_layer"])
    wm_after = UchidaWatermark.from_state_dict(attacked_layer, ckpt["watermark_state"])

    # Measure metrics
    _, test_loader = DATA_REGISTRY[ckpt.get("dataset")](batch_size=256)
    _, attacked_test_acc = evaluate(attacked_model, test_loader, device)

    wm_retention = watermark_retention(wm_after.bit_accuracy())
    acc_retention = task_accuracy_retention(attacked_test_acc, baseline_test_acc)

    print(f"\nAttack: {cfg['attack']}")
    print(
        f"  watermark_retention:      {wm_retention:.3f}  (1.0 = fully intact, ~0.5 = destroyed)"
    )
    print(
        f"  task_accuracy_retention:  {acc_retention:.3f}  (attacked_acc={attacked_test_acc:.4f} / baseline_acc={baseline_test_acc:.4f})"
    )

    # Log and save
    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled"):
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        with mlflow.start_run(run_name=mlflow_cfg["run_name"]):
            mlflow.log_params(
                {
                    "model": "resnet18_cifar",
                    "watermark_scheme": "uchida",
                    "attack": cfg["attack"],
                    **{k: v for k, v in cfg.items() if k not in ("mlflow",)},
                }
            )
            mlflow.log_metrics(
                {
                    "watermark_retention": wm_retention,
                    "task_accuracy_retention": acc_retention,
                    "attacked_test_acc": attacked_test_acc,
                }
            )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    out_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_name"])
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": cfg["model"],
            "dataset": cfg["dataset"],
            "model_state_dict": attacked_model.state_dict(),
            "attack": cfg["attack"],
            "watermark_retention": wm_retention,
            "task_accuracy_retention": acc_retention,
        },
        out_path,
    )
    print(f"\nSaved attacked checkpoint to {out_path}")


if __name__ == "__main__":
    main()
