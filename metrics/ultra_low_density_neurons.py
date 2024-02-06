import torch
from torchmetrics import Metric
import wandb
import numpy as np
import pandas as pd


class UltraLowDensityNeurons(Metric):
    def __init__(self, n_features: int, return_neuron_indices=False, threshold=1e-6, dist_sync_on_step=False):
        """
        :param n_features: number of features in the layer
        :param return_neuron_indices: whether to return the indices of the ultra-low freq neurons
            if True, the compute method will return a tensor with the indices of the ultra-low freq neurons
            if False, the compute method will return the fraction of ultra-low freq neurons
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_features = n_features
        self.return_neuron_indices = return_neuron_indices
        self.threshold = threshold

        self.add_state("num_active", default=torch.zeros(n_features, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, features: torch.Tensor):
        self.num_active += (features > 1e-7).sum(dim=0)
        self.total += features.shape[0]

    def compute(self, return_wandb_bar=False):
        freqs = self.num_active.float() / self.total
        if return_wandb_bar:
            table = wandb.Table(dataframe=pd.DataFrame({"freqs": torch.log10(freqs).cpu()}))
            return wandb.plot.histogram(table, "freqs")
        if self.return_neuron_indices:
            return (freqs < self.threshold).nonzero(as_tuple=True)[0]
        return (freqs < self.threshold).sum() / self.n_features
