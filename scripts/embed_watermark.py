"""
Embedding Uchida-style watermark on an already trained baseline models

Consider this as a short fine tuning task where we load the baseline checkpoint 
weights. Then continue training with the watermark
regularization loss added on top of ordinary cross-entropy, for a small
number of epochs. This is the standard way Uchida-style watermarks are
embedded in practice.

L(total) = L(training) + A*L(watermark)


Run command : python scripts/embed_watermark.py --config configs/watermarks/uchida_resnet.yaml
"""

import argparse
import os
import sys  # used for file path and directories
import time
 
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # find the project root and adds into the projects import path

from src.models.resnet_cifar import resnet18_cifar
from src.models.lenet5 import lenet5

from src.data.loaders import get_cifar_loaders
from src.data.loaders import get_mnist_loaders

from src.training.loop import train_one_epoch, evaluate
from src.watermarks.uchida import UchidaWatermark
from src.utils.model_utils import get_layer_by_name
from src.utils.path_resolver import resolve_path

MODEL_REGISTRY = {
    "resnet18_cifar": resnet18_cifar,
    "lenet5": lenet5,
}
 
DATA_REGISTRY = {
    "cifar10": get_cifar_loaders,
    "mnist": get_mnist_loaders,
}

# Watermarking process
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args() # now args.config will contain "--config configs/watermarks/uchida.yaml"
    
    # Finds the YAML file and converts the YAML key-value to dictionary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = resolve_path(args.config, project_root)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
        
    torch.manual_seed(cfg["seed_train"]) # for reproducibility during fine tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Load the trained baseline
    baseline_path = resolve_path(cfg["baseline_checkpoint"], project_root)
    baseline = torch.load(baseline_path, map_location=device)
    baseline_acc = baseline["test_acc"]
    print(f"Loaded baseline checkpoint -- baseline test_acc = {baseline_acc:.4f}")
    
    model = MODEL_REGISTRY[cfg["model"]](num_classes=cfg["num_classes"]).to(device) # Creating a new ResNet-18 Note: At this moment the weights are randomly initialzed
    model.load_state_dict(baseline["model_state_dict"]) # replace random weights with baseline weights.
    # Note: Now "model" is a clean trained ResNet
    
    # Setting watermark on chosen layer
    target_layer = get_layer_by_name(model, cfg["watermark_layer"])
    # To check if the selected layer is actually a Conv
    assert isinstance(target_layer, torch.nn.Conv2d), \
        f"'{cfg['watermark_layer']}' is not a Conv2d layer -- check the name against `print(model)`"
    wm = UchidaWatermark(target_layer, message_len=cfg["message_len"], seed=cfg["seed"])
    print(f"Watermark initialized on layer '{cfg['watermark_layer']}' "
          f"-- initial bit accuracy (should be ~0.5, i.e. unembedded): {wm.bit_accuracy():.3f}")

    # Fine-tune with the watermark loss added in 
    train_loader, test_loader = DATA_REGISTRY[cfg["dataset"]](batch_size=cfg["batch_size"])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                                 weight_decay=cfg["weight_decay"], nesterov=True)
 
    lambda_wm = cfg["lambda_wm"]
    extra_loss_fn = lambda m: lambda_wm * wm.regularization_loss(m)
    
    mlflow_cfg = cfg.get("mlflow", {})
    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        mlflow.start_run(run_name=mlflow_cfg["run_name"])
        mlflow.log_params({
            "model": cfg["model"], "watermark_scheme": "uchida", "attack": "none",
            "watermark_layer": cfg["watermark_layer"], "message_len": cfg["message_len"],
            "lambda_wm": lambda_wm, "epochs": cfg["epochs"],
        })
        
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device,
                                                    extra_loss_fn=extra_loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, device)
        bit_acc = wm.bit_accuracy()
        elapsed = time.time() - t0
        
        print(f"epoch {epoch:2d}/{cfg['epochs']} | "
            f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f} | "
            f"watermark_bit_acc {bit_acc:.3f} | {elapsed:.1f}s")
        
        if mlflow_cfg.get("enabled"):
            import mlflow
            mlflow.log_metrics({"test_acc": test_acc, "watermark_bit_acc": bit_acc}, step=epoch)
            
        if bit_acc == 1.0 and epoch >= 3:
            # Once the watermark is fully embedded and a few epochs have
            # passed to let accuracy stabilize around it, no need to keep
            # fine-tuning further, extra epochs mostly risk accuracy drift.
            print("Watermark fully embedded (bit_acc = 1.0) -- stopping early.")
            break
            
    # Save watermarked checkpoint, including the key needed to re-verify later 
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    checkpoint_path = os.path.join(cfg["checkpoint_dir"], cfg["checkpoint_name"])
    torch.save({
        "model_state_dict": model.state_dict(), 
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "test_acc": test_acc,
        "baseline_test_acc": baseline_acc,
        "watermark_scheme": "uchida",
        "watermark_layer": cfg["watermark_layer"],
        "watermark_state": wm.state_dict(), # Saves both X(key) and b(message) allong with the message length. You cannot verify wihtout these values
    }, checkpoint_path)
    
    print(f"\nSaved watermarked checkpoint to {checkpoint_path}")
    print(f"  test_acc: {test_acc:.4f}  (baseline was {baseline_acc:.4f} -- "
          f"accuracy_retention so far = {test_acc / baseline_acc:.4f})")
    print(f"  watermark_bit_acc: {wm.bit_accuracy():.3f}")
    
    if mlflow_cfg.get("enabled"):
        import mlflow
        mlflow.log_metric("final_accuracy_retention", test_acc / baseline_acc)
        mlflow.log_artifact(checkpoint_path)
        mlflow.end_run()
    
if __name__ == "__main__":
    main()   