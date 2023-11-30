from typing import Tuple

import torch
from torch import nn
from transformer_lens.hook_points import HookedRootModule, HookPoint


class HookedSparseAutoencoder(HookedRootModule):

    def __init__(self, d_input, d_hidden, *args):
        super().__init__(*args)
        self.hook_sae_in = HookPoint()
        self.hook_sae_pre = HookPoint()
        self.hook_sae_post = HookPoint()
        self.hook_sae_out = HookPoint()

        self.d_hidden = d_hidden

        self.encoder_map = nn.Linear(d_input, d_hidden)
        self.nonlinearity = nn.ReLU()
        self.decoder_map = nn.Linear(d_hidden, d_input, bias=False)
        self.tied_bias = nn.Parameter(torch.zeros(d_input))

        self.setup()

    def get_weights(self):
        W_enc = self.encoder_map.weight
        b_enc = self.encoder_map.bias
        W_dec = self.decoder_map.weight
        b_tied = self.tied_bias
        return W_enc, b_enc, W_dec, b_tied

    def set_weights(self, W_enc, b_enc, W_dec, b_tied):
        self.encoder_map.weight = W_enc
        self.encoder_map.bias = b_enc
        self.decoder_map.weight = W_dec
        self.tied_bias = b_tied

    def encoder(self, X: torch.Tensor) -> torch.Tensor:
        X = self.hook_sae_in(X)
        X = X - self.tied_bias
        X = self.encoder_map(X)
        X = self.hook_sae_pre(X)
        X = self.nonlinearity(X)
        feature_activations = self.hook_sae_post(X)
        return feature_activations

    def decoder(self, feature_activations: torch.Tensor) -> torch.Tensor:
        X = self.decoder_map(feature_activations) + self.tied_bias
        X = self.hook_sae_out(X)
        return X

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_activations = self.encoder(X)
        X = self.decoder(feature_activations)
        return X, feature_activations

    @torch.no_grad()
    def make_decoder_weights_and_grad_unit_norm(self):
        W_dec_normed = self.decoder_map.weight.data / self.decoder_map.weight.data.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.decoder_map.weight.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.decoder_map.weight.grad -= W_dec_grad_proj
        # Bugfix(?) for ensuring W_dec retains unit norm, this was not there when I trained my original autoencoders.
        self.decoder_map.weight.data = W_dec_normed

    def make_decoder_weights_unit_norm(self):
        W_dec_normed = self.decoder_map.weight.data / self.decoder_map.weight.data.norm(dim=-1, keepdim=True)
        self.decoder_map.weight.data = W_dec_normed
