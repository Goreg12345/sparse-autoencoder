import torch
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from transformer_lens import HookedTransformer


class DeadFeatureResampler(Metric):
    def __init__(self, llm: HookedTransformer, text_dataset: IterableDataset, sae_component, num_samples,
                 n_features,
                 **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.text_dataset = iter(text_dataset)
        self.losses = None
        self.buffer = None  # initialized later
        self.actv_size = None  # computed from cache
        self.actv_name = sae_component
        self.num_samples = num_samples
        self.n_features = n_features

        self.add_state("dead_neurons", default=torch.zeros(n_features, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, features: torch.Tensor):
        self.dead_neurons += (features > 0.).sum(dim=0)

    @torch.no_grad()
    def compute_losses(self):

        for i in range(1000000):
            batch = next(self.text_dataset)
            # only take half of it because the size of the batch might be optimized for the activation extraction
            # and now we have to feedforward the whole network which consumes more memory
            batch = batch[:16]
            batch[:, 0] = self.llm.tokenizer.bos_token_id
            names_filter = lambda name_: self.actv_name in name_
            batch = batch.to(self.llm.W_E.device)
            token_loss, cache = self.llm.run_with_cache(batch, names_filter=names_filter, return_type='loss',
                                                        loss_per_token=True)
            if i == 0:
                self.actv_size = cache[self.actv_name].shape[-1]
                self.buffer = torch.empty((self.num_samples, self.actv_size), dtype=torch.float32, device='cpu')
                self.losses = torch.empty((self.num_samples,), dtype=torch.float32, device='cpu')
            acts = cache[self.actv_name]
            acts = acts[:, :-1, :]  # remove the last token for which we don't have a loss
            acts = acts.reshape(-1, self.actv_size)
            token_loss = token_loss.view(-1)

            # always -1 because the last token has no loss
            start = i * acts.shape[0]
            end = start + acts.shape[0]
            if end > self.buffer.shape[0]:
                end = self.buffer.shape[0]
                acts = acts[:end - start]
                token_loss = token_loss[:end - start]
            self.buffer[start:end] = acts.to('cpu')
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

        self.compute_losses()
        idx = torch.multinomial(self.losses, idx_dead.numel())
        new_directions = self.buffer[idx].to(W_enc.device)

        # Renormalize the input vector to have unit L2 norm and set this to be the dictionary vector (W_dec)
        # for the dead autoencoder neuron.
        W_dec.data[:, idx_dead] = (new_directions / new_directions.norm(dim=-1, keepdim=True)).T

        # For the corresponding encoder vector, renormalize the input vector to equal the average norm of the encoder
        # weights for alive neurons × 0.2.
        W_enc.data[idx_dead] = W_enc[idx_dead] / W_enc[idx_dead].norm(dim=-1, keepdim=True) * 0.2 * W_enc[~idx_dead].norm(dim=-1).mean()

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
