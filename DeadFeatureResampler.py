import torch
from torch import nn
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from transformer_lens import HookedTransformer

from HookedSparseAutoencoder import HookedSparseAutoencoder


class DeadFeatureResampler(Metric):
    def __init__(self, sae: HookedSparseAutoencoder, loader, num_samples,
                 actv_size,
                 n_features,
                 **kwargs):
        super().__init__(**kwargs)
        self.sae = sae
        self.loader = loader
        self.losses = None
        self.buffer = None  # initialized later
        self.num_samples = num_samples
        self.n_features = n_features
        self.actv_size = actv_size

        self.add_state("dead_neurons", default=torch.zeros(n_features, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, features: torch.Tensor):
        self.dead_neurons += (features > 1e-6).sum(dim=0)

    @torch.no_grad()
    def compute_losses(self, loader):
        self.buffer = torch.empty((self.num_samples, self.actv_size), dtype=torch.float32, device='cpu')
        self.losses = torch.empty((self.num_samples,), dtype=torch.float32, device='cpu')

        for i, X in enumerate(loader):
            X = X.to(self.sae.decoder_map.weight.device)
            recons, features = self.sae(X)
            token_loss = nn.functional.mse_loss(X, recons, reduction='none').mean(dim=-1)

            start = i * X.shape[0]
            end = start + X.shape[0]
            if end > self.buffer.shape[0]:
                end = self.buffer.shape[0]
                X = X[:end - start]
                token_loss = token_loss[:end - start]
            self.buffer[start:end] = X.to('cpu')
            self.losses[start:end] = token_loss.to('cpu')

            # buffer is full
            if end == self.buffer.shape[0]:
                break

        # Assign each input vector a probability of being picked that is proportional to the square of the
        # autoencoder’s loss on that input.
        self.losses = self.losses ** 2
        self.losses /= self.losses.sum()

    def compute(self, W_enc, b_enc, W_dec, b_tied, optimizer):
        idx_dead = (self.dead_neurons == 0.).nonzero(as_tuple=True)[0]
        print('Resampling, dead neurons:', idx_dead.numel())

        if len(idx_dead) == 0:
            return W_enc, b_enc, W_dec, b_tied

        self.compute_losses(self.loader)
        idx = torch.multinomial(self.losses, idx_dead.numel())
        new_directions = self.buffer[idx].to(W_enc.device)

        # Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector (W_dec)
        # for the dead autoencoder neuron.
        W_dec.data[:, idx_dead] = (new_directions / new_directions.norm(dim=-1, keepdim=True)).T

        # For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder
        # weights for alive neurons × 0.2.
        W_enc.data[idx_dead] = new_directions / new_directions.norm(dim=-1, keepdim=True) * 0.2 * W_enc[~idx_dead].norm(dim=-1).mean()

        # Set the corresponding encoder bias element to zero.
        b_enc.data[idx_dead] = 0.

        # reset optimizer state at dead features
        w_dec_state = optimizer.state[W_dec]
        w_dec_state['exp_avg'][:, idx_dead] = 0.
        w_dec_state['exp_avg_sq'][:, idx_dead] = 0.

        w_enc_state = optimizer.state[W_enc]
        w_enc_state['exp_avg'][idx_dead] = 0.
        w_enc_state['exp_avg_sq'][idx_dead] = 0.

        b_enc_state = optimizer.state[b_enc]
        b_enc_state['exp_avg'][idx_dead] = 0.
        b_enc_state['exp_avg_sq'][idx_dead] = 0.

        return W_enc, b_enc, W_dec, b_tied
