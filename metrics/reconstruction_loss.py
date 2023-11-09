import torch
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from transformer_lens import HookedTransformer

from HookedSparseAutoencoder import HookedSparseAutoencoder


class ReconstructionLoss(Metric):
    def __init__(self, llm: HookedTransformer, sae: HookedSparseAutoencoder, text_dataset: IterableDataset,
                 sae_component, **kwargs):
        super().__init__(**kwargs)

        self.llm = llm
        self.sae = sae
        self.text_dataset = iter(text_dataset)
        self.sae_component = sae_component

        self.add_state("loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("reconstruction_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("zero_ablation_loss", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def replacement_hook(self, sae_input, hook):
        actvs_reconstr = self.sae(sae_input)[0]  # [0] is reconstruction, [1] are latent features
        return actvs_reconstr

    def mean_ablate_hook(self, sae_input, hook):
        sae_input[:] = sae_input.mean([0, 1])
        return sae_input

    def zero_ablate_hook(self, sae_input, hook):
        sae_input[:] = 0.
        return sae_input

    def update(self):
        batch = next(self.text_dataset)

        loss = self.llm(batch, return_type="loss")
        recons_loss = self.llm.run_with_hooks(batch, return_type="loss", fwd_hooks=[
            (self.sae_component, self.replacement_hook)])
        zero_abl_loss = self.llm.run_with_hooks(batch, return_type="loss",
                                             fwd_hooks=[(self.sae_component, self.mean_ablate_hook)])
        self.loss += loss
        self.reconstruction_loss += recons_loss
        self.zero_ablation_loss += zero_abl_loss
        self.total += 1

    def compute(self):
        loss = self.loss / self.total
        recons_loss = self.reconstruction_loss / self.total
        zero_abl_loss = self.zero_ablation_loss / self.total

        score = ((zero_abl_loss - recons_loss) / (zero_abl_loss - loss))
        return score
