from typing import Any, Union, Optional, Callable, IO, Iterable, List

import wandb
from transformer_lens import HookedTransformer
from typing_extensions import Self
import pandas as pd

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn, Tensor
from torch.optim import Optimizer

from DeadFeatureResampler import DeadFeatureResampler
from SparseAutoencoder import SparseAutoencoder
from metrics.L0Loss import L0Loss
from metrics.dead_neurons import DeadNeurons
from metrics.reconstruction_loss import ReconstructionLoss
from metrics.ultra_low_density_neurons import UltraLowDensityNeurons


class SAETrainer(pl.LightningModule):

    def __init__(self, sae: SparseAutoencoder, resampling_steps: List, n_resampling_watch_steps,
                 l1_coefficient: float = 6e-3, reconstruction_loss_metric_zero=None,
                 reconstruction_loss_metric_mean=None,
                 dead_features_resampler: DeadFeatureResampler = None, lr=1e-3,
                 l1_scheduler: Callable = None, lr_warmup_steps = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sae = sae
        self.l1_coefficient = l1_coefficient
        self.reconstruction_loss_metric_zero = reconstruction_loss_metric_zero
        self.reconstruction_loss_metric_mean = reconstruction_loss_metric_mean
        self.dead_neurons_metric = DeadNeurons(sae.d_hidden)
        self.l0_loss = L0Loss()
        self.low_freq_metric = UltraLowDensityNeurons(sae.d_hidden)
        self.frequency_hist = UltraLowDensityNeurons(sae.d_hidden)
        self.lr = lr
        self.l1_scheduler = l1_scheduler

        # this one will be used to track dead neurons for the neuron resampling method
        self.resampling_steps = resampling_steps
        self.n_resampling_watch_steps = n_resampling_watch_steps
        self.dead_features_resampler = dead_features_resampler

        # learning rate warmup
        if lr_warmup_steps is not None:
            self.lr_warmup_steps = lr_warmup_steps

    def forward(self, X):
        return self.sae(X)

    def criterion(self, X_hat, X, feature_activations):
        mse = nn.functional.mse_loss(X_hat, X, reduction='mean')
        l1 = self.l1_coefficient * (feature_activations.abs()**1).sum(dim=1).mean()
        return mse + l1, mse, l1

    def training_step(self, batch, batch_idx):
        X = batch
        if batch_idx == 0:
            if self.sae.cfg['init_geometric_median']:
                self.sae.init_geometric_median(X)

        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mse", mse, prog_bar=True, logger=True)
        self.log("train_l1", l1, prog_bar=True, logger=True)

        self.dead_neurons_metric.update(feature_activations)
        if batch_idx % 2000 == 0:
            self.log("dead_neurons", self.dead_neurons_metric.compute())
            self.dead_neurons_metric.reset()

        if batch_idx > 1000:  # wait until the model has been trained for a bit
            self.low_freq_metric.update(feature_activations)
            self.frequency_hist.update(feature_activations)
            if batch_idx % 2000 == 0:
                self.log("low_freq_neurons", self.low_freq_metric.compute())
                self.low_freq_metric.reset()
                self.logger.experiment.log({"frequency_hist": self.frequency_hist.compute(return_wandb_bar=True)})
                self.frequency_hist.reset()

        if self.dead_features_resampler:
            # if batch_idx between any of resampling steps - 1000 and resampling steps
            if any([batch_idx in range(step - self.n_resampling_watch_steps, step) for step in self.resampling_steps]):
                self.dead_features_resampler.update(feature_activations)
            if batch_idx in self.resampling_steps:
                # this resamples and sets new weights of self.sae
                self.dead_features_resampler.compute(self.sae, self.trainer.optimizers[0])
                self.dead_features_resampler.reset()

        if self.l1_scheduler:
            self.l1_coefficient = self.l1_scheduler(batch_idx)

        return loss.float()

    def validation_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mse", mse, prog_bar=True, logger=True)
        self.log("val_l1", l1, prog_bar=True, logger=True)
        if self.reconstruction_loss_metric_zero:
            self.reconstruction_loss_metric_zero.update()
        if self.reconstruction_loss_metric_mean:
            self.reconstruction_loss_metric_mean.update()
        self.l0_loss.update(feature_activations)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.reconstruction_loss_metric_zero:
            self.log("val_reconstruction_loss_zero", self.reconstruction_loss_metric_zero.compute(), logger=True)
            self.reconstruction_loss_metric_zero.reset()
        if self.reconstruction_loss_metric_mean:
            self.log("val_reconstruction_loss_mean", self.reconstruction_loss_metric_mean.compute(), logger=True)
            self.reconstruction_loss_metric_mean.reset()
        self.log("val_l0_loss", self.l0_loss.compute(), logger=True)
        self.l0_loss.reset()

    def on_test_epoch_end(self) -> None:
        self.on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        if hasattr(self, 'lr_warmup_steps'):
            def get_lr_multiplier(step):
                if step < self.lr_warmup_steps:
                    return step / self.lr_warmup_steps
                elif any([step >= resampling_step and step < resampling_step + self.lr_warmup_steps
                          for resampling_step in self.resampling_steps]):
                    return ((step % self.lr_warmup_steps) / self.lr_warmup_steps)
                return 1.
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier),
                'interval': 'step',
            }
            return [optimizer], [scheduler]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        self.sae.make_decoder_weights_and_grad_unit_norm()

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.sae.make_decoder_weights_unit_norm()

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        cfg,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> Self:
        sae = SparseAutoencoder.load(checkpoint_path, cfg)
        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, sae=sae,
                                             resampling_steps=cfg['resampling_steps'],
                                             n_resampling_watch_steps=cfg['n_resampling_watch_steps'],
                                             **kwargs)
        return model
