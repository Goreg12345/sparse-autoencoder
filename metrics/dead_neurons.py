import torch
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
