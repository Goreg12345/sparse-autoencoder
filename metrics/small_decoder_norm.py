import torch
from torchmetrics import Metric
import wandb
import numpy as np
import pandas as pd
import pytorch_lightning as pl


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


class SmallDecoderNormCallback(pl.Callback):
    def __init__(self, n_features, return_neuron_indices=False, threshold=0.99, log_interval=5000):
        super().__init__()
        self.metric = SmallDecoderNorm(n_features, return_neuron_indices, threshold)
        self.hist_metric = SmallDecoderNorm(n_features, return_neuron_indices, threshold)
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.metric.to(pl_module.device)
        self.hist_metric.to(pl_module.device)

        self.metric.update(pl_module.sae.W_dec)

        # Log the histogram at specified intervals
        if batch_idx % self.log_interval == 0:
            self.hist_metric.update(pl_module.sae.W_dec)
            histogram = self.hist_metric.compute(return_wandb_bar=True)
            trainer.logger.experiment.log({"decoder_norm_hist": histogram})
            self.hist_metric.reset()
            trainer.logger.experiment.log({"small_decoder_norm": self.metric.compute().item()})
            self.metric.reset()
