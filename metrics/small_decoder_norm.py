import torch
from torchmetrics import Metric
import wandb
import numpy as np
import pandas as pd


class SmallDecoderNorm(Metric):
    def __init__(
        self,
        n_features: int,
        return_neuron_indices=False,
        threshold=0.99,
        dist_sync_on_step=False,
    ):
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

        self.add_state(
            "dec_norms",
            default=torch.zeros(n_features, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum"
        )

    def update(self, W_dec: torch.Tensor):
        norm = W_dec.data.norm(dim=-1, keepdim=False)
        self.dec_norms += norm
        self.total += 1

    def compute(self, return_wandb_bar=False):
        norms = self.dec_norms.float() / self.total
        if return_wandb_bar:
            table = wandb.Table(dataframe=pd.DataFrame({"dec_norms": norms.cpu()}))
            return wandb.plot.histogram(table, "dec_norms")
        if self.return_neuron_indices:
            return (norms < self.threshold).nonzero(as_tuple=True)[0]
        return (norms < self.threshold).sum() / self.n_features
