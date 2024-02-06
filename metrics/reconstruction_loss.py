import torch
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from transformer_lens import HookedTransformer

from HookedSparseAutoencoder import HookedSparseAutoencoder


class ReconstructionLoss(Metric):
    def __init__(self, llm: HookedTransformer, sae: HookedSparseAutoencoder, text_dataset: IterableDataset,
                 sae_component, head=None, ablation_type='zero', **kwargs):
        super().__init__(**kwargs)

        self.llm = llm
        self.sae = sae
        self.text_dataset = iter(text_dataset)
        self.sae_component = sae_component
        self.head = head
        if ablation_type == 'zero':
            self.ablation_hook = self.zero_ablate_hook
        elif ablation_type == 'mean':
            self.ablation_hook = self.mean_ablate_hook

        self.add_state("reconstruction_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def replacement_hook(self, sae_input, hook):
        shape = sae_input.shape
        if self.head is not None:
            if self.head == 'concat':
                sae_input_proc = sae_input.reshape(-1, sae_input.shape[-1] * sae_input.shape[-2])
            else:
                sae_input_proc = sae_input[..., self.head, :].reshape(-1, shape[-1])
        else:
            sae_input_proc = sae_input.view(-1, shape[-1])
        actvs_reconstr = self.sae(sae_input_proc)[0]  # [0] is reconstruction, [1] are latent features

        # if we only use one head, only replace the activations of that head
        if self.head is not None and type(self.head) == int:
            shape = list(shape)
            del shape[-2]
            actvs_reconstr = actvs_reconstr.view(shape)
            sae_input[:, :, self.head, :] = actvs_reconstr
            return sae_input
        actvs_reconstr = actvs_reconstr.view(shape)
        return actvs_reconstr

    def mean_ablate_hook(self, sae_input, hook):
        if type(self.head) == int:  # if we only use one head, don't average over the rest of the heads
            sae_input[:, :, self.head, :] = sae_input[:, :, self.head, :].mean([0, 1], keepdim=True)
            return sae_input

        sae_input[:] = sae_input.mean([0, 1], keepdim=True)
        return sae_input

    def zero_ablate_hook(self, sae_input, hook):
        if type(self.head) == int:
            sae_input[:, :, self.head, :] = 0.
            return sae_input
        sae_input[:] = 0.
        return sae_input

    def update(self):
        batch = next(self.text_dataset)
        # insert bos token
        batch[:, 0] = self.llm.tokenizer.bos_token_id
        loss = self.llm(batch, return_type="loss")
        recons_loss = self.llm.run_with_hooks(batch, return_type="loss", fwd_hooks=[
            (self.sae_component, self.replacement_hook)])
        zero_abl_loss = self.llm.run_with_hooks(batch, return_type="loss",
                                             fwd_hooks=[(self.sae_component, self.ablation_hook)])
        self.reconstruction_score += ((zero_abl_loss - recons_loss) / (zero_abl_loss - loss))
        self.total += 1

    def compute(self):
        return self.reconstruction_score / self.total
