from typing import Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_input, d_hidden, cfg=None, *args):
        super().__init__(*args)

        self.d_hidden = d_hidden

        self.cfg = cfg

        self.W_enc = nn.Parameter(torch.empty(d_input, d_hidden))
        self.b_enc = nn.Parameter(torch.empty(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_input))
        self.b_dec = nn.Parameter(torch.empty(d_input))
        self.mean = nn.Parameter(torch.empty(d_input), requires_grad=False)
        self.standard_norm = nn.Parameter(
            torch.tensor(1, dtype=torch.float32), requires_grad=False
        )
        # initialize
        self.reset_parameters()

    def reset_parameters(self):
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
        if self.cfg.get("adjust_for_dict_size", False):
            self.standard_norm.data = self.standard_norm.data * torch.sqrt(
                torch.tensor(self.d_hidden, dtype=torch.float32)
            )

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        # subtract decoder bias
        if not self.cfg.get("disable_decoder_bias", False):
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
        if self.cfg["allow_lower_decoder_norm"]:
            norm_greater_than_one = (norm > 1).view(-1)
            self.W_dec.data[norm_greater_than_one] = (
                self.W_dec.data[norm_greater_than_one] / norm[norm_greater_than_one]
            )
        else:
            self.W_dec.data = self.W_dec.data / norm

    @classmethod
    def load(self, path, cfg):
        sae = SparseAutoencoder(cfg["actv_size"], cfg["d_hidden"], cfg=cfg)
        checkpoint = torch.load(path)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        # remove sae. prefix
        state_dict = {k.replace("sae.", ""): v for k, v in state_dict.items()}
        if "mean" not in state_dict:
            state_dict["mean"] = torch.zeros(cfg["actv_size"])
        if "standard_norm" not in state_dict:
            state_dict["standard_norm"] = torch.tensor(1, dtype=torch.float32)
        sae.load_state_dict(state_dict)
        return sae


class GatedSparseAutoencoder(SparseAutoencoder):
    def __init__(self, d_input, d_hidden, cfg=None, *args):
        super().__init__(d_input, d_hidden, cfg=cfg, *args)

        self.r_mag = nn.Parameter(torch.empty(d_hidden))
        self.b_gate = nn.Parameter(torch.empty(d_hidden))

    def encoder(self, X: torch.Tensor, training=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        # subtract decoder bias
        if not self.cfg.get("disable_decoder_bias", False):
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

    def forward(self, X: torch.Tensor, training=True) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if training:
            feature_activations, pi_gate = self.encoder(X, training)
            X_recons = self.decoder(feature_activations)
            return X_recons, feature_activations, pi_gate
        else:
            feature_activations = self.encoder(X, training)
            X = self.decoder(feature_activations)
            return X, feature_activations

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.zeros_(self.r_mag)  # e^r would be 1, so it initially doesn't change the magnitude
        nn.init.zeros_(self.b_gate)