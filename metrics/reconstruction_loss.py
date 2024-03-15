import torch
from torch.utils.data import IterableDataset
from torchmetrics import Metric
from transformer_lens import HookedTransformer

from SparseAutoencoder import SparseAutoencoder


class ReconstructionLoss(Metric):
    def __init__(
        self,
        llm: HookedTransformer,
        sae: SparseAutoencoder,
        text_dataset: IterableDataset,
        sae_component,
        head=None,
        ablation_type="zero",
        seq_position="all",
        **kwargs
    ):
        # seq_position: if 'all', the whole sequence is used for the reconstruction loss, if tensor with indices,
        # only the indices are used for the reconstruction loss
        super().__init__(**kwargs)

        self.llm = llm
        self.sae = sae
        self.text_dataset = iter(text_dataset)
        self.sae_component = sae_component
        self.head = head
        if ablation_type == "zero":
            self.ablation_hook = self.zero_ablate_hook
        elif ablation_type == "mean":
            self.ablation_hook = self.mean_ablate_hook

        # assert seq_position == 'all' or (type(seq_position) == torch.Tensor and all([type(i) == torch.int32 for i in seq_position]))
        assert seq_position == "all" or seq_position.shape != torch.Size([])
        self.seq_position = seq_position

        # self.add_state("reconstruction_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("zero_abl_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("recons_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def replacement_hook(self, sae_input, hook):
        shape = sae_input.shape
        if self.head is not None:
            if self.head == "concat":
                sae_input_proc = sae_input.reshape(
                    -1, sae_input.shape[-1] * sae_input.shape[-2]
                )
            else:
                sae_input_proc = sae_input[..., self.head, :].reshape(-1, shape[-1])
        else:
            sae_input_proc = sae_input.view(-1, shape[-1])
        actvs_reconstr = self.sae(sae_input_proc)[
            0
        ]  # [0] is reconstruction, [1] are latent features

        # if we only use one head, only replace the activations of that head
        if self.head is not None and type(self.head) == int:
            shape = list(shape)
            del shape[-2]
            actvs_reconstr = actvs_reconstr.view(shape)
            sae_input[:, :, self.head, :] = actvs_reconstr
            return sae_input
        actvs_reconstr = actvs_reconstr.view(shape)
        return actvs_reconstr

    def mean_ablate_hook(self, sae_input, hook):  # possible problem: it's not the real mean if the batch is small; it will be yield a lower reconstruction score than it should
        if (
            type(self.head) == int
        ):  # if we only use one head, don't average over the rest of the heads
            sae_input[:, :, self.head, :] = sae_input[:, :, self.head, :].mean(
                [0, 1], keepdim=True
            )
            return sae_input

        sae_input[:] = sae_input.mean([0, 1], keepdim=True)
        return sae_input

    def zero_ablate_hook(self, sae_input, hook):
        if type(self.head) == int:
            sae_input[:, :, self.head, :] = 0.0
            return sae_input
        sae_input[:] = 0.0
        return sae_input

    def update(self):
        batch = next(self.text_dataset)
        # insert bos token
        batch[:, 0] = self.llm.tokenizer.bos_token_id
        batch = batch.to(self.llm.W_E.device)
        if self.seq_position == "all":
            loss = self.llm(batch, return_type="loss")
            recons_loss = self.llm.run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(self.sae_component, self.replacement_hook)],
            )
            zero_abl_loss = self.llm.run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(self.sae_component, self.ablation_hook)],
            )
        else:
            loss = self.llm(batch, return_type="loss", loss_per_token=True)
            recons_loss = self.llm.run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(self.sae_component, self.replacement_hook)],
                loss_per_token=True,
            )
            zero_abl_loss = self.llm.run_with_hooks(
                batch,
                return_type="loss",
                fwd_hooks=[(self.sae_component, self.ablation_hook)],
                loss_per_token=True,
            )
            loss = loss[:, self.seq_position].mean()
            recons_loss = recons_loss[:, self.seq_position].mean()
            zero_abl_loss = zero_abl_loss[:, self.seq_position].mean()
        loss = loss.to(self.loss.device)
        recons_loss = recons_loss.to(self.loss.device)
        zero_abl_loss = zero_abl_loss.to(self.loss.device)
        # print(f"loss: {loss}, recons_loss: {recons_loss}, zero_abl_loss: {zero_abl_loss}", f'reconstruction_score: {(zero_abl_loss - recons_loss) / (zero_abl_loss - loss)}')

        # self.reconstruction_score += ((zero_abl_loss - recons_loss) / (zero_abl_loss - loss))
        self.zero_abl_loss += zero_abl_loss
        self.recons_loss += recons_loss
        self.loss += loss
        self.total += 1

    def compute(self):
        return (self.zero_abl_loss / self.total - self.recons_loss / self.total) / (
            self.zero_abl_loss / self.total - self.loss / self.total
        )
        # return self.reconstruction_score / self.total

    # Don't return any parameters, otherwise they will be saved in the checkpoint
    # or tracked by the optimizer
    def parameters(self, recurse: bool = True):
        return iter([])

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return iter([])

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return {}
