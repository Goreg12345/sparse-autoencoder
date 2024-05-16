import torch
import pytorch_lightning as pl
from torchmetrics import Metric


class DeadNeurons(Metric):
    def __init__(
        self, n_features: int, return_neuron_indices=False, dist_sync_on_step=False
    ):
        """
        :param n_features: number of features in the layer
        :param return_neuron_indices: whether to return the indices of the dead neurons
            if True, the compute method will return a tensor with the indices of the dead neurons
            if False, the compute method will return the fraction of dead neurons
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_features = n_features
        self.return_neuron_indices = return_neuron_indices

        self.add_state(
            "dead_neurons",
            default=torch.zeros(n_features, dtype=torch.int),
            dist_reduce_fx="sum",
        )

    def update(self, features: torch.Tensor):
        self.dead_neurons += (features > 0.0).sum(dim=0)

    def compute(self):
        if self.return_neuron_indices:
            return (self.dead_neurons == 0.0).nonzero(as_tuple=True)[0]
        return (self.dead_neurons == 0.0).sum() / self.n_features


class DeadNeuronsCallback(pl.Callback):
    def __init__(self, n_features, return_neuron_indices=False, log_interval=2000):
        super().__init__()
        self.metric = DeadNeurons(n_features, return_neuron_indices)
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Extract the feature activations from the training_step output
        self.metric.to(pl_module.device)
        feature_activations = outputs['feature_activations']
        self.metric.update(feature_activations)

        if batch_idx % self.log_interval == 0:
            trainer.logger.experiment.log({'dead_neurons': self.metric.compute().item()})
            self.metric.reset()
