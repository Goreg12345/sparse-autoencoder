import torch
from torchmetrics import Metric
import wandb
import pandas as pd
import pytorch_lightning as pl


class UltraLowDensityNeurons(Metric):
    def __init__(
        self,
        n_features: int,
        return_neuron_indices=False,
        threshold=1e-6,
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
            "num_active",
            default=torch.zeros(n_features, dtype=torch.int),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total", default=torch.zeros(1, dtype=torch.int), dist_reduce_fx="sum"
        )

    def update(self, features: torch.Tensor):
        self.num_active += (features > 1e-7).sum(dim=0)
        self.total += features.shape[0]

    def compute(self, return_wandb_bar=False):
        freqs = self.num_active.float() / self.total
        if return_wandb_bar:
            table = wandb.Table(
                dataframe=pd.DataFrame({"freqs": torch.log10(freqs).cpu()})
            )
            return wandb.plot.histogram(table, "freqs")
        if self.return_neuron_indices:
            return (freqs < self.threshold).nonzero(as_tuple=True)[0]
        return (freqs < self.threshold).sum() / self.n_features


class UltraLowDensityNeuronsCallback(pl.Callback):
    def __init__(self, n_features, return_neuron_indices=False, threshold=1e-6, log_interval=2000,
                 start_batch_idx=1000):
        super().__init__()
        self.metric = UltraLowDensityNeurons(n_features, return_neuron_indices, threshold)
        self.hist_metric = UltraLowDensityNeurons(n_features, return_neuron_indices, threshold)
        self.log_interval = log_interval
        self.start_batch_idx = start_batch_idx

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.metric.to(pl_module.device)
        self.hist_metric.to(pl_module.device)

        if batch_idx > self.start_batch_idx:
            feature_activations = outputs['feature_activations']
            self.metric.update(feature_activations)
            self.hist_metric.update(feature_activations)

            # Log the metrics at specified intervals
            if batch_idx % self.log_interval == 0:
                trainer.logger.experiment.log({"low_freq_neurons": self.metric.compute().item()})
                self.metric.reset()

                histogram = self.hist_metric.compute(return_wandb_bar=True)
                trainer.logger.experiment.log({"frequency_hist": histogram})
                self.hist_metric.reset()
