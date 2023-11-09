from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn, Tensor

from DeadFeatureResampler import DeadFeatureResampler
from HookedSparseAutoencoder import HookedSparseAutoencoder
from metrics.L0Loss import L0Loss
from metrics.dead_neurons import DeadNeurons


class SparseAutoencoder(pl.LightningModule):

    def __init__(self, sae: HookedSparseAutoencoder, l1_coefficient: float = 6e-3, reconstruction_loss_metric=None,
                 dead_features_resampler: DeadFeatureResampler = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sae = sae
        self.l1_coefficient = l1_coefficient
        self.reconstruction_loss_metric = reconstruction_loss_metric
        self.dead_neurons_metric = DeadNeurons(sae.d_hidden)
        self.l0_loss = L0Loss()

        # this one will be used to track dead neurons for the neuron resampling method
        self.resampling_steps = [25000, 50000, 75000, 100000, 150000, 200000, 250000, 300000]
        self.dead_features_resampler = dead_features_resampler

    def forward(self, X):
        return self.sae(X)

    def criterion(self, X_hat, X, feature_activations):
        mse = nn.functional.mse_loss(X_hat, X, reduction='mean')
        l1 = self.l1_coefficient * feature_activations.abs().sum(dim=1).mean()
        return mse + l1, mse, l1

    def training_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mse", mse, prog_bar=True, logger=True)
        self.log("train_l1", l1, prog_bar=True, logger=True)
        self.dead_neurons_metric.update(feature_activations)
        if batch_idx % 1000 == 0:
            self.log("dead_neurons", self.dead_neurons_metric.compute())
            self.dead_neurons_metric.reset()

        if self.dead_features_resampler:
            # if batch_idx between any of resampling steps - 1000 and resampling steps
            if any([batch_idx in range(step - 3500, step) for step in self.resampling_steps]):
                self.dead_features_resampler.update(feature_activations)
            if batch_idx in self.resampling_steps:
                w = self.dead_features_resampler.compute(*self.sae.get_weights(), self.trainer.optimizers[0])
                self.dead_features_resampler.reset()
                self.sae.set_weights(*w)

        return loss.float()

    def validation_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mse", mse, prog_bar=True, logger=True)
        self.log("val_l1", l1, prog_bar=True, logger=True)
        if self.reconstruction_loss_metric:
            self.reconstruction_loss_metric.update()
        self.l0_loss.update(feature_activations)
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.reconstruction_loss_metric:
            self.log("val_reconstruction_loss", self.reconstruction_loss_metric.compute(), logger=True)
            self.reconstruction_loss_metric.reset()
        self.log("val_l0_loss", self.l0_loss.compute(), logger=True)

    def on_test_epoch_end(self) -> None:
        if self.reconstruction_loss_metric:
            self.log("test_reconstruction_loss", self.reconstruction_loss_metric.compute(), logger=True)
            self.reconstruction_loss_metric.reset()

    def test_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_mse", mse, prog_bar=True, logger=True)
        self.log("test_l1", l1, prog_bar=True, logger=True)
        if self.reconstruction_loss_metric:
            self.reconstruction_loss_metric.update()
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        self.sae.make_decoder_weights_and_grad_unit_norm()