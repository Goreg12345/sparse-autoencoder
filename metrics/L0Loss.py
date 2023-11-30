import torch
from torchmetrics import Metric


class L0Loss(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("l0_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, features: torch.Tensor):
        self.l0_loss += (features > 0.).sum()
        self.total += features.shape[0]

    def compute(self):
        return self.l0_loss.float() / self.total