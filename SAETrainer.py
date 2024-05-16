from typing import Any, Union, Optional, Callable, IO

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch import nn, Tensor
from torch.optim import Optimizer
from typing_extensions import Self

from SparseAutoencoder import SparseAutoencoder, GatedSparseAutoencoder
from training.DeadFeatureResampler import DeadFeatureResampler
from training.config import SAEConfig


class SAETrainer(pl.LightningModule):
    def __init__(
        self,
        config: SAEConfig,
        sae: SparseAutoencoder,
        dead_features_resampler: DeadFeatureResampler = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.config = config
        self.sae = sae
        self.dead_features_resampler = dead_features_resampler

        self.l1_scheduler = None

        # this one will be used to track dead neurons for the neuron resampling method
        #self.dead_features_resampler = dead_features_resampler
        self.dead_features_resampler = None
        self.kwargs = kwargs

    def forward(self, X):
        return self.sae(X)

    def criterion(self, X_hat, X, feature_activations):
        # mse = nn.functional.mse_loss(X_hat, X, reduction='mean')
        # for mse use standardized activations and reconstructions because l1 is applied to standardized activations
        # this helps to achieve same l0 for different layers
        X = (X - self.sae.mean) / self.sae.standard_norm
        X_hat = (X_hat - self.sae.mean) / self.sae.standard_norm
        if self.config.adjust_for_dict_size:
            mse = ((X - X_hat) ** 2).sum(dim=1).mean()
        else:
            mse = nn.functional.mse_loss(X_hat, X, reduction="mean")
        l1 = (
            self.config.l1_coefficient
            * (feature_activations.abs() ** self.config.l1_exponent).sum(dim=1).mean()
        )
        return mse + l1, mse, l1

    def training_step(self, batch, batch_idx):
        X = batch
        if batch_idx == 0:
            if self.config.init_geometric_median:
                self.sae.init_geometric_median(X)

        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_mse", mse, prog_bar=True, logger=True)
        self.log("train_l1", l1, prog_bar=True, logger=True)

        if self.dead_features_resampler:
            # if batch_idx between any of resampling steps - 1000 and resampling steps
            if any(
                [
                    batch_idx in range(step - self.n_resampling_watch_steps, step)
                    for step in self.resampling_steps
                ]
            ):
                self.dead_features_resampler.update(feature_activations)
            if batch_idx in self.resampling_steps:
                # this resamples and sets new weights of self.sae
                self.dead_features_resampler.compute(
                    self.sae, self.trainer.optimizers[0]
                )
                self.dead_features_resampler.reset()

        if self.l1_scheduler:
            self.config.l1_coefficient = self.l1_scheduler(batch_idx, self.config.l1_coefficient)

        # return multiple values such that metrics can use them through callbacks
        return {
            'loss': loss.float(),
            'feature_activations': feature_activations,
            'reconstructions': X_hat,
        }

    def validation_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_mse", mse, prog_bar=True, logger=True)
        self.log("val_l1", l1, prog_bar=True, logger=True)
        return {
            'loss': loss.float(),
            'feature_activations': feature_activations,
            'reconstructions': X_hat,
        }

    def test_step(self, batch, batch_idx):
        X = batch
        X_hat, feature_activations = self(X)
        loss, mse, l1 = self.criterion(X_hat, X, feature_activations)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        self.log("test_mse", mse, prog_bar=True, logger=True)
        self.log("test_l1", l1, prog_bar=True, logger=True)
        return {
            'loss': loss.float(),
            'feature_activations': feature_activations,
            'reconstructions': X_hat,
        }

    def configure_optimizers(self):
        beta_1 = self.config.beta1
        beta_2 = self.config.beta2
        start_lr_decay = self.config.start_lr_decay
        end_lr_decay = self.config.end_lr_decay
        if hasattr(self.config, "lr_warmup_steps"):

            def get_lr_multiplier(step):
                if step < self.config.lr_warmup_steps:
                    return step / self.config.lr_warmup_steps
                elif any(
                    [
                        step >= resampling_step
                        and step < resampling_step + self.config.lr_warmup_steps
                        for resampling_step in self.config.resampling_steps
                    ]
                ):
                    return (step % self.config.lr_warmup_steps) / self.config.lr_warmup_steps
                elif step > start_lr_decay and step < end_lr_decay:
                    return 1 - (step - start_lr_decay) / (end_lr_decay - start_lr_decay)
                return 1.0

            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.config.lr, betas=(beta_1, beta_2)
            )
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(
                    optimizer, get_lr_multiplier
                ),
                "interval": "step",
            }
            return [optimizer], [scheduler]
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.lr, betas=(beta_1, beta_2)
        )
        return optimizer

    def backward(self, loss: Tensor, *args: Any, **kwargs: Any) -> None:
        loss.backward(*args, **kwargs)
        self.sae.make_grad_unit_norm()

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
        model = super().load_from_checkpoint(
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            sae=sae,
            resampling_steps=cfg["resampling_steps"],
            n_resampling_watch_steps=cfg["n_resampling_watch_steps"],
            **kwargs,
        )
        return model


class GatedSAETrainer(SAETrainer):
    ...

    def init_sae(self):
        self.sae = GatedSparseAutoencoder(self.config)
        if self.config.init_geometric_median:
            self.sae.init_geometric_median(torch.tensor(self.buffer.buffer[:10000000]))
        if self.config.standardize_activations:
            self.sae.init_activation_standardization(
                torch.tensor(self.buffer.buffer[:10000000])
            )