import csv
from typing import Literal
import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

from circuits import Node
from training.config import SAEConfig
from training.train import train


def make_config(sae_type: Literal["vanilla", "gated"], l1_coefficient: float, node: Node):
    actv_name = node.actv_name
    layer = node.layer
    head = node.head
    kwargs = {
        "use_disc_dataset": True,
        "init_geometric_median": True,
        "disable_decoder_bias": True,
        "beta1": 0.0,
        "beta2": 0.999,
        "lr": 0.0003,
        "train_steps": 30000,
        "target_metric": "test_reconstruction_loss_mean"
    }
    return SAEConfig(sae_type=sae_type, l1_coefficient=l1_coefficient,
                        actv_name=actv_name, layer=layer, head=head, **kwargs)


def optimize_l1_coefficient(model, target, low=1e-6, high=1e2, tolerance=1e-4, factor=10.0, max_iterations=10):
    best_param = (low * high) ** 0.5  # Initial best guess
    best_diff = abs(model(best_param) - target)

    for _ in range(max_iterations):
        mid = (low * high) ** 0.5  # Geometric mean for exponential adjustments
        model_output = model(mid)
        diff = abs(model_output - target)

        if diff < best_diff:
            best_param = mid
            best_diff = diff

        if diff < tolerance:
            return mid

        if model_output > target:
            low = mid / factor  # Adjust lower bound exponentially
        else:
            high = mid * factor  # Adjust upper bound exponentially

    return best_param  # Best approximation of the parameter after max_iterations


def logarithmic_search(target_metric=0.8, output_file='search_results.csv'):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['SAE Type', 'Node', 'Best L1 Coefficient', 'Result'])

        sae_type: Literal["vanilla", "gated"]
        for sae_type in ["gated"]:
            for node in [Node('z', 9, 9, seq_pos='end')]:
                def model(l1_coefficient):
                    config = make_config(sae_type, l1_coefficient, node)
                    return train(config)

                best_l1 = optimize_l1_coefficient(model, target_metric)
                best_result = model(best_l1)

                writer.writerow([sae_type, f'{node}', best_l1, best_result])
                file.flush()  # Ensure the data is written to the file immediately


if __name__ == "__main__":
    logarithmic_search()
