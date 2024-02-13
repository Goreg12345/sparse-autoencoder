from typing import Tuple

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
        self.standard_norm = nn.Parameter(torch.tensor(1, dtype=torch.float32), requires_grad=False)
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
        X = (acts - self.mean) / self.standard_norm
        self.b_dec.data = X.mean(dim=0)

    def init_activation_standardization(self, acts):
        self.mean.data = acts.mean(dim=0)
        self.standard_norm.data = acts.norm(dim=1).mean()

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        # standardize input
        X = (X - self.mean) / self.standard_norm
        # subtract decoder bias
        X = X - self.b_dec
        X = X @ self.W_enc + self.b_enc # batch d_input, d_input d_hidden
        return F.relu(X)

    def decoder(self, feature_activations: torch.Tensor) -> torch.Tensor:
        recons = feature_activations @ self.W_dec + self.b_dec
        return recons * self.standard_norm + self.mean

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_activations = self.encoder(X)
        X = self.decoder(feature_activations)
        return X, feature_activations

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.W_dec.data = W_dec_normed

    def make_decoder_weights_unit_norm(self):
        W_dec_normed = self.W_dec.data / self.W_dec.data.norm(dim=-1, keepdim=True)
        self.W_dec.data = W_dec_normed

    @classmethod
    def load(self, path, cfg):
      sae = SparseAutoencoder(cfg['actv_size'], cfg['d_hidden'], cfg=cfg)
      checkpoint = torch.load(path)
      state_dict = checkpoint['state_dict']
      # remove sae. prefix
      state_dict = {k.replace('sae.', ''): v for k, v in state_dict.items()}
      sae.load_state_dict(state_dict)
      return sae
