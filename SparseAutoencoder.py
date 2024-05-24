from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from activation_buffer import Buffer
from training.config import SAEConfig


def sae_from_config(config: SAEConfig, buffer: Buffer = None) -> "SparseAutoencoder":
    if config.sae_type == "vanilla":
        sae = SparseAutoencoder(config)
    elif config.sae_type == "gated":
        sae = GatedSparseAutoencoder(config)
    elif config.sae_type == "anthropic":
        sae = AnthropicSAE(config)
    else:
        raise ValueError(f"Unknown SAE type: {config.sae_type}")
    sae.reset_parameters(buffer)
    return sae


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SAEConfig, *args):
        super().__init__(*args)

        d_hidden = config.d_hidden
        d_input = config.actv_size
        self.cfg = config

        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.b_enc = nn.Parameter(torch.empty(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_dec = nn.Parameter(torch.empty(d_input))
        self.mean = nn.Parameter(torch.empty(d_input), requires_grad=False)
        self.standard_norm = nn.Parameter(
            torch.tensor(1, dtype=torch.float32), requires_grad=False
        )

    def reset_parameters(self, buffer: Buffer = None):
        # we don't need to initialize the decoder weights with kaiming initialization,
        # because we normalize them to unit norm anyways
        nn.init.uniform_(self.W_dec, -1, 1)
        # normalize
        self.W_dec.data /= self.W_dec.data.norm(dim=-1, keepdim=True)
        # although encoder and decoder weights are not tied, we initialize them to be the same as a good starting point
        nn.init.uniform_(self.W_enc, -1, 1)
        # normalize
        self.W_enc.data /= self.W_enc.data.norm(dim=-1, keepdim=True)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        nn.init.zeros_(self.mean)
        nn.init.ones_(self.standard_norm)

        if buffer is not None:
            acts = buffer.buffer[:10000000]
            if self.cfg.init_geometric_median:
                self.init_geometric_median(acts)
            if self.cfg.standardize_activations:
                self.init_activation_standardization(acts)

    def init_geometric_median(self, acts):
        # standardize input
        # X = (acts - self.mean) / self.standard_norm
        # self.b_dec.data = X.mean(dim=0)
        self.mean.data = acts.median(dim=0)[
            0
        ]  # median returns a tuple of (values, indices)

    def init_activation_standardization(self, acts):
        acts = acts - self.mean
        self.standard_norm.data = acts.norm(dim=1).mean()
        if self.cfg.adjust_for_dict_size:
            self.standard_norm.data = self.standard_norm.data * torch.sqrt(
                torch.tensor(self.d_hidden, dtype=torch.float32)
            )

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        # subtract decoder bias
        if not self.cfg.disable_decoder_bias:
            X = X - self.b_dec
        X = X @ self.W_enc + self.b_enc  # batch d_input, d_input d_hidden
        return F.relu(X)

    def decoder(self, feature_activations: torch.Tensor) -> torch.Tensor:
        recons = feature_activations @ self.W_dec + self.b_dec
        return recons * self.standard_norm + self.mean

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_activations = self.encoder(X)
        X = self.decoder(feature_activations)
        return X, feature_activations

    @torch.no_grad()
    def make_grad_unit_norm(self):
        W_dec_normed = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj

        # self.W_dec.data = W_dec_normed

    def make_decoder_weights_unit_norm(self):
        norm = self.W_dec.data.norm(dim=-1, keepdim=True)
        if self.cfg.allow_lower_decoder_norm:
            norm_greater_than_one = (norm > 1).view(-1)
            self.W_dec.data[norm_greater_than_one] = (
                self.W_dec.data[norm_greater_than_one] / norm[norm_greater_than_one]
            )
        else:
            self.W_dec.data = self.W_dec.data / norm

    @classmethod
    def load(self, path, cfg: SAEConfig):
        sae = SparseAutoencoder(config=cfg)
        checkpoint = torch.load(path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        # remove sae. prefix
        state_dict = {k.replace("sae.", ""): v for k, v in state_dict.items()}
        if "mean" not in state_dict:
            state_dict["mean"] = torch.zeros(cfg.actv_size)
        if "standard_norm" not in state_dict:
            state_dict["standard_norm"] = torch.tensor(1, dtype=torch.float32)
        sae.load_state_dict(state_dict)
        return sae

    def create_trainer(self, loader: torch.utils.data.DataLoader = None):
        from SAETrainer import SAETrainer
        if self.cfg.resampling_steps:
            if loader is None:
                raise ValueError("loader must be provided for resampling")
            from training.DeadFeatureResampler import DeadFeatureResampler
            resampler = DeadFeatureResampler(
                loader, self.cfg.n_resampler_samples, self.cfg.actv_size, self.cfg.d_hidden
            )
            return SAETrainer(self.cfg, self, dead_features_resampler=resampler)
        return SAETrainer(self.cfg, self)


class GatedSparseAutoencoder(SparseAutoencoder):
    def __init__(self, config: SAEConfig, *args):
        super().__init__(config, *args)

        d_hidden = config.d_hidden
        self.r_mag = nn.Parameter(torch.empty(d_hidden))
        self.b_gate = nn.Parameter(torch.empty(d_hidden))

    def encoder(self, X: torch.Tensor, training=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        # subtract decoder bias
        if not self.cfg.disable_decoder_bias:
            X = X - self.b_dec
        X = X @ self.W_enc # batch d_input, d_input d_hidden
        X_mag = X * torch.exp(self.r_mag) + self.b_enc
        X_mag = F.relu(X_mag)

        pi_gate = X + self.b_gate
        # binarize
        X_gate = (pi_gate > 0).float()
        if training:
            return X_mag * X_gate, pi_gate

        return X_mag * X_gate

    def forward(self, X: torch.Tensor, training=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if training:
            feature_activations, pi_gate = self.encoder(X, training)
            X_recons = self.decoder(feature_activations)
            return X_recons, feature_activations, pi_gate
        else:
            feature_activations = self.encoder(X, training)
            X = self.decoder(feature_activations)
            return X, feature_activations

    def reset_parameters(self, buffer: Buffer = None):
        super().reset_parameters(buffer)
        nn.init.zeros_(self.r_mag)  # e^r would be 1, so it initially doesn't change the magnitude
        nn.init.zeros_(self.b_gate)

    def create_trainer(self, loader: torch.utils.data.DataLoader = None):
        from SAETrainer import GatedSAETrainer
        if self.cfg.resampling_steps:
            if loader is None:
                raise ValueError("loader must be provided for resampling")
            from training.DeadFeatureResampler import DeadFeatureResampler
            resampler = DeadFeatureResampler(
                loader, self.cfg.n_resampler_samples, self.cfg.actv_size, self.cfg.d_hidden
            )
            return GatedSAETrainer(self.cfg, self, dead_features_resampler=resampler)
        return GatedSAETrainer(self.cfg, self)


class AnthropicSAE(SparseAutoencoder):
    def __init__(self, config: SAEConfig, *args):
        super().__init__(config, *args)

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        X = X @ self.W_enc + self.b_enc
        return F.relu(X)

    def reset_parameters(self, buffer: Buffer = None):
        super().reset_parameters(buffer)
        # make W_dec columns have norm 0.1
        self.W_dec.data *= 0.1
        # init W_enc as W_dec^T as per anthropic blog post
        self.W_enc.data = self.W_dec.data.T

    def clip_gradient_norm(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)

    def create_trainer(self, loader: torch.utils.data.DataLoader = None):
        from SAETrainer import AnthropicSAETrainer
        if self.cfg.resampling_steps:
            if loader is None:
                raise ValueError("loader must be provided for resampling")
            from training.DeadFeatureResampler import DeadFeatureResampler
            resampler = DeadFeatureResampler(
                loader, self.cfg.n_resampler_samples, self.cfg.actv_size, self.cfg.d_hidden
            )
            return AnthropicSAETrainer(self.cfg, self, dead_features_resampler=resampler)
        return AnthropicSAETrainer(self.cfg, self)

    def init_activation_standardization(self, acts):
        acts = torch.tensor(acts, dtype=torch.float32, device=self.W_enc.device)
        acts = acts - self.mean
        n = self.cfg.actv_size
        sqrt_n = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=acts.device))
        # goal: E[||X||] = sqrt(n)
        self.standard_norm.data = acts.norm(dim=1).mean() / sqrt_n
        if self.cfg.adjust_for_dict_size:
            self.standard_norm.data = self.standard_norm.data * torch.sqrt(
                torch.tensor(self.d_hidden, dtype=torch.float32)
            )