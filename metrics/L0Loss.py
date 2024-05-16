import torch
from torchmetrics import Metric
import pytorch_lightning as pl


class L0Loss(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("l0_loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, features: torch.Tensor):
        self.l0_loss += (features > 1e-8).sum()
        self.total += features.shape[0]

    def compute(self):
        return self.l0_loss.float() / self.total


class L0LossCallback(pl.Callback):
    def __init__(self, log_interval=2000, calc_on="validation"):
        super().__init__()
        self.metric = L0Loss()
        self.log_interval = log_interval
        self.calc_on = calc_on

    def on_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, phase):
        # Ensure the metric is on the correct device
        self.metric.to(pl_module.device)

        # Extract the feature activations from the step output
        feature_activations = outputs['feature_activations']
        self.metric.update(feature_activations)

        if phase == self.calc_on and batch_idx % self.log_interval == 0:
            trainer.logger.experiment.log({f'{phase}_l0_loss': self.metric.compute().item()})
            self.metric.reset()

    def on_epoch_end(self, trainer, pl_module, phase):
        # Ensure the metric is on the correct device
        self.metric.to(pl_module.device)

        if phase == self.calc_on:
            trainer.logger.experiment.log({f'{phase}_l0_loss': self.metric.compute().item()})
            self.metric.reset()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "training")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.on_batch_end(trainer, pl_module, outputs, batch, batch_idx, "validation")

    def on_train_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, "training")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, "validation")
