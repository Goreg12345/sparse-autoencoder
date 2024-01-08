from typing import Any, Union, Optional, Callable, IO, Iterable, List

from transformer_lens import HookedTransformer
from typing_extensions import Self

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn, Tensor
from torch.optim import Optimizer

from DeadFeatureResampler import DeadFeatureResampler
from HookedSparseAutoencoder import HookedSparseAutoencoder
from metrics.L0Loss import L0Loss
from metrics.dead_neurons import DeadNeurons
from metrics.reconstruction_loss import ReconstructionLoss
from metrics.ultra_low_density_neurons import UltraLowDensityNeurons


class SparseAutoencoder(pl.LightningModule):

    def __init__(self, sae: HookedSparseAutoencoder, resampling_steps: List, n_resampling_watch_steps,
                 l1_coefficient: float = 6e-3, reconstruction_loss_metric=None,
                 dead_features_resampler: DeadFeatureResampler = None, lr=1e-3,
                 l1_scheduler: Callable = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sae = sae
        self.l1_coefficient = l1_coefficient
        self.reconstruction_loss_metric = reconstruction_loss_metric
        self.dead_neurons_metric = DeadNeurons(sae.d_hidden)
        self.l0_loss = L0Loss()
        self.low_freq_metric = UltraLowDensityNeurons(sae.d_hidden)
        self.lr = lr
        self.l1_scheduler = l1_scheduler

        # this one will be used to track dead neurons for the neuron resampling method
        self.resampling_steps = resampling_steps
        self.n_resampling_watch_steps = n_resampling_watch_steps
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

        if batch_idx > 2000:  # wait until the model has been trained for a bit
            self.low_freq_metric.update(feature_activations)
            if batch_idx % 10000 == 0:
                self.log("low_freq_neurons", self.low_freq_metric.compute())
                self.low_freq_metric.reset()

        if self.dead_features_resampler:
            # if batch_idx between any of resampling steps - 1000 and resampling steps
            if any([batch_idx in range(step - self.n_resampling_watch_steps, step) for step in self.resampling_steps]):
                self.dead_features_resampler.update(feature_activations)
            if batch_idx in self.resampling_steps:
                w = self.dead_features_resampler.compute(*self.sae.get_weights(), self.trainer.optimizers[0])
                self.dead_features_resampler.reset()
                self.sae.set_weights(*w)

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

    def on_save_checkpoint(self, checkpoint):
        if 'reconstruction_loss_metric' in checkpoint:
            del checkpoint['reconstruction_loss_metric']
        return checkpoint

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        llm: HookedTransformer,
        text_dataset: Iterable,
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Self:
        sae = HookedSparseAutoencoder(768, 8*768)
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint['state_dict']
        # filter for sae.
        state_dict = {k: v for k, v in state_dict.items() if k.startswith('sae')}
        # remove sae. prefix
        state_dict = {k[4:]: v for k, v in state_dict.items()}
        sae.load_state_dict(state_dict)

        # TODO: this is a hack to get the reconstruction loss metric to load
        reconstruction_loss_metric = ReconstructionLoss(llm, sae, text_dataset, None)
        dead_features_resampler = DeadFeatureResampler(sae, None, 200000, 768, 8*768)
        model = super().load_from_checkpoint(checkpoint_path, map_location, hparams_file, strict, sae=sae,
                                             reconstruction_loss_metric=reconstruction_loss_metric,
                                             dead_features_resampler=dead_features_resampler, **kwargs)
        return model

    @property
    def W_enc(self):
        return self.sae.encoder_map.weight

    @property
    def b_enc(self):
        return self.sae.encoder_map.bias

    @property
    def W_dec(self):
        return self.sae.decoder_map.weight

    @property
    def b_tied(self):
        return self.sae.tied_bias