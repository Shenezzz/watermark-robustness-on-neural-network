"""
Train a clean (non-watermarked) baseline model from a config file.

Usage:
    python scripts/train_baseline.py --config configs/models/resnet18_cifar10.yaml
    python scripts/train_baseline.py --config configs/models/lenet5_mnist.yaml

Run from the project root (dnn-watermark-robustness/) so the src.* imports
resolve, or install the project with `pip install -e .` for a proper package.
"""

import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root, for `src.*` imports

from src.models.resnet18 import resnet18_cifar
from src.models.lenet5 import lenet5_mnist

from src.data.loaders import get_cifar_loaders
from src.data.loaders import get_mnist_loaders

from src.training.loop import train_one_epoch, evaluate


MODEL_REGISTRY = {
    "resnet18_cifar": resnet18_cifar,
    "lenet5": lenet5_mnist
}

DATA_REGISTRY = {
    "cifar10": get_cifar_loaders,
    "mnist": get_mnist_loaders
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True) # create a directory, if already existing dont throw an error 
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_name"])

    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        mlflow.start_run(run_name=mlflow_cfg["run_name"])
        mlflow.log_params({
            "model": cfg["model"], "watermark_scheme": "none", "attack": "none",
            "epochs": cfg["epochs"], "batch_size": cfg["batch_size"], "lr": cfg["lr"],
        })

    train_loader, test_loader = DATA_REGISTRY[cfg["dataset"]](batch_size=cfg["batch_size"])
    
    model = MODEL_REGISTRY[cfg["model"]](num_classes=cfg["num_classes"]).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                 weight_decay=cfg["weight_decay"], nesterov=True)
    
    # to gradually change the learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_acc = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device) # total_loss / total, correct / total
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"epoch {epoch:3d}/{cfg['epochs']} | "
              f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
              f"test_loss {test_loss:.4f} test_acc {test_acc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.5f} | {elapsed:.1f}s")

        if mlflow_cfg.get("enabled"):
            import mlflow
            mlflow.log_metrics({
                "train_loss": train_loss, "train_acc": train_acc,
                "test_loss": test_loss, "test_acc": test_acc,
            }, step=epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({"model_state_dict": model.state_dict(), "test_acc": test_acc, "epoch": epoch},
                       checkpoint_path)

    print(f"\nBest test accuracy: {best_acc:.4f} -- checkpoint saved to {checkpoint_path}")
    print("This test_acc is the reference number for every later "
          "task_accuracy_retention calculation (retention = attacked_acc / this_acc).")

    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.log_metric("best_test_acc", best_acc)
        mlflow.log_artifact(checkpoint_path)
        mlflow.end_run()

if __name__ == "__main__":
    main()