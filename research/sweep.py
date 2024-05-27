import argparse
import csv
from typing import Literal
import os

import fire

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_num', type=int, default=0, help='GPU number to use')
parser.add_argument('--target_metric', type=float, default=0.8, help='Target metric to optimize'
                    )
parser.add_argument('--output_file', type=str, default='search_results.csv', help='Output file to save the results')
parser.add_argument('--component_name', type=str, default='z', help='Component name to optimize')
parser.add_argument('--head', type=int, default=9, help='Head to optimize')
parser.add_argument('--layer', type=int, default=9, help='Layer to optimize')
args = parser.parse_args()

# Set CUDA_VISIBLE_DEVICES environment variable
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

import sys
sys.path.append('/tmp/pycharm_project_451')


from circuits import Node
from training.config import SAEConfig
from training.train import train


def make_config(sae_type: Literal["vanilla", "gated", "anthropic", "l0_5"], l1_coefficient: float, node: Node):
    actv_name = node.actv_name
    layer = node.layer
    head = node.head
    wandb_name = f"{actv_name}_l{layer}h{head}_{sae_type}_l1_{l1_coefficient}"
    if sae_type == "vanilla":
        kwargs = {
            "use_disc_dataset": True,
            "init_geometric_median": False,
            "disable_decoder_bias": False,
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": 0.0003,
            "train_steps": 30000,
            "target_metric": "test_reconstruction_loss_mean",
            "store_size": 30000 * 2048,
            "wandb_name": wandb_name,
        }
    elif sae_type == "gated":
        kwargs = {
            "use_disc_dataset": True,
            "init_geometric_median": False,
            "disable_decoder_bias": False,
            "beta1": 0.0,
            "beta2": 0.999,
            "lr": 0.0003,
            "train_steps": 30000,
            "target_metric": "test_reconstruction_loss_mean",
            "store_size": 30000 * 2048,
            "wandb_name": wandb_name,
        }
    elif sae_type == "anthropic":
       kwargs = {
            "use_disc_dataset": True,
            "disable_decoder_bias": True,
            "sae_type": "anthropic",
            "beta1": 0.9,
            "beta2": 0.999,
            "train_steps": 30000,
            "target_metric": "test_reconstruction_loss_mean",
            "start_lr_decay": 26000,
            "end_lr_decay": 30000,
            "lr": 0.00005,
            "store_size": 30000 * 2048,
            "wandb_name": wandb_name,
       }
    elif sae_type == "l0_5":
        kwargs = {
            "use_disc_dataset": True,
            "disable_decoder_bias": False,
            "sae_type": "l0_5",
            "beta1": 0.0,
            "beta2": 0.999,
            "train_steps": 30000,
            "target_metric": "test_reconstruction_loss_mean",
            "l1_exponent": 0.5,
            "lr": 0.0003,
            "store_size": 30000 * 2048,
            "wandb_name": wandb_name,
        }
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")
    return SAEConfig(sae_type=sae_type, l1_coefficient=l1_coefficient,
                        actv_name=actv_name, layer=layer, head=head, **kwargs)


def optimize_l1_coefficient(model, target, low=1e-6, high=1e2, tolerance=1e-4, factor=10.0, max_iterations=10):
    best_param = (low * high) ** 0.5  # Initial best guess
    best_result = model(best_param)
    best_diff = abs(best_result - target)

    for _ in range(max_iterations):
        mid = (low * high) ** 0.5  # Geometric mean for exponential adjustments
        model_output = model(mid)
        diff = abs(model_output - target)

        if diff < best_diff:
            best_param = mid
            best_diff = diff
            best_result = model_output

        if diff < tolerance:
            return mid, best_result

        if model_output > target:
            low = mid / factor  # Adjust lower bound exponentially
        else:
            high = mid * factor  # Adjust upper bound exponentially

    return best_param, best_result  # Best approximation of the parameter after max_iterations


def logarithmic_search(target_metric=0.8, output_file='search_results.csv',
                       component_name='z', head=9, layer=9):

    sae_type: Literal["vanilla", "gated"]
    for sae_type in ["vanilla", "gated", "l0_5", "anthropic"]:
        node = Node(component_name=component_name, head=head, layer=layer)

        def model(l1_coefficient):
            config = make_config(sae_type, l1_coefficient, node)
            return train(config)

        best_l1, best_result = optimize_l1_coefficient(model, target_metric)

        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([sae_type, node.component_name, node.layer, node.head, best_l1, best_result])
            file.flush()  # Ensure the data is written to the file immediately

if __name__ == "__main__":
    fire.Fire(logarithmic_search)
