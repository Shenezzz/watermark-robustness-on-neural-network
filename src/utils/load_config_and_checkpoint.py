import argparse
import os
import yaml
import torch

from src.utils.path_resolver import resolve_path

def load_config_and_checkpoint(test_acc_keyword):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Finds the YAML file and converts the YAML key-value to dictionary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = resolve_path(args.config, project_root)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = resolve_path(
        cfg["watermarked_checkpoint"], project_root
    )

    ckpt = torch.load(ckpt_path, map_location=device)

    baseline_test_acc = ckpt[test_acc_keyword]

    return cfg, device, ckpt, baseline_test_acc