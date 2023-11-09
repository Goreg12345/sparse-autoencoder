import os

from DeadFeatureResampler import DeadFeatureResampler

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import einops
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

from HookedSparseAutoencoder import HookedSparseAutoencoder
from SparseAutoencoder import SparseAutoencoder
from metrics.reconstruction_loss import ReconstructionLoss
from text_dataset import TextDataset


class Buffer(IterableDataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, llm, dataset, buffer_size, actv_size, seq_len, buffer_batches, extraction_batch_size, actv_name, layer, batch_size ):
        self.buffer = torch.zeros((int(buffer_size), actv_size), requires_grad=False).to('cpu')
        self.token_pointer = 0
        self.first = True
        self.llm = llm
        self.buffer_batches = buffer_batches
        self.extraction_batch_size = extraction_batch_size
        self.actv_name = actv_name
        self.layer = layer
        self.actv_size = actv_size
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.dataset = dataset
        print('setting up now')
        self.refresh()

    def add_batch_to_buffer(self, batch):
        batch[:, 0] = self.llm.tokenizer.bos_token_id
        names_filter = lambda name_: self.actv_name in name_
        batch = batch.to(self.llm.W_E.device)
        _, cache = self.llm.run_with_cache(batch, stop_at_layer=self.layer + 1, names_filter=names_filter)
        acts = cache[self.actv_name].reshape(-1, self.actv_size)

        # if the buffer is close to full, we might not be able to fit the whole batch in
        end = self.pointer + acts.shape[0] if self.pointer + acts.shape[0] < self.buffer.shape[0] else self.buffer.shape[0]
        self.buffer[self.pointer:end] = acts[:end - self.pointer]
        self.pointer = end
        print('new batch added to buffer')

    @torch.no_grad()
    def refresh(self):
        # TODO: if last batch is too small, need to add it to the buffer and then resize the buffer
        self.pointer = 0
        while True:
            batch = next(self.dataset)
            # if batch is smaller than extraction batch size, it's the last batch and we need to resize the buffer
            if batch.shape[0] < self.extraction_batch_size:
                self.add_batch_to_buffer(batch)
                self.buffer = self.buffer[:self.pointer]
                break
            # if buffer is full, stop
            if self.pointer >= self.buffer.shape[0]:
                break
            self.add_batch_to_buffer(batch)

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to('cpu')]

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        out = self.buffer[self.pointer:self.pointer + self.batch_size]
        self.pointer += self.batch_size
        if self.pointer > self.buffer.shape[0] - self.batch_size:
            # print("Refreshing the buffer!")
            self.refresh()
        return out


if __name__=='__main__':
    import torch
    import datasets
    # %%
    from datasets import load_dataset

    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
    # %%
    cfg = {
        'actv_size': 768,
        'buffer_batches': 100,
        'buffer_size': 1e7,
        'extraction_batch_size': 500,
        'actv_name': 'blocks.8.hook_resid_post',
        'layer': 8,
        'batch_size': 2048,
        'seq_len': 512
    }
    # %%
    from transformer_lens import HookedTransformer

    llm = HookedTransformer.from_pretrained(
        model_name='gpt2-small',
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
        device='cpu'  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    text_dataset = TextDataset(dataset, llm.to_tokens, cfg['extraction_batch_size'], drop_last_batch=False,
                               seq_len=cfg['seq_len'])
    # don't increase num_workers > 1 because it's an IterableDataset and multiple dataloaders will yield the same data
    text_dataset_loader = iter(DataLoader(text_dataset, batch_size=None, shuffle=False, num_workers=1, prefetch_factor=200))
    next(text_dataset_loader)
    buffer = Buffer(
        llm.cuda(),
        text_dataset_loader,
        **cfg
    )
    loader = torch.utils.data.DataLoader(buffer, batch_size=None, shuffle=False, num_workers=0)

    d_hidden = cfg['actv_size'] * 8
    sae = HookedSparseAutoencoder(d_input=cfg['actv_size'], d_hidden=d_hidden)
    reconstruction_loss_metric = ReconstructionLoss(llm, sae,
                                                    TextDataset(dataset, llm.to_tokens, 16,
                                                                drop_last_batch=False,
                                                                seq_len=cfg['seq_len'])
                                                    , cfg['actv_name'])

    resampler = DeadFeatureResampler(llm, text_dataset, cfg['actv_name'], 200000, d_hidden)

    wandb_logger = pl.loggers.WandbLogger(project='serimats', config=cfg,
                                          name=f'sae_debug')
    model = SparseAutoencoder(sae, 1e-2, reconstruction_loss_metric, resampler)
    # load pretrained model
    #model = SparseAutoencoder.load_from_checkpoint('models/sae_resampleDebug_22itcheck_l1=1e-2_lr=2e-4_redpajama_8x.ckpt')
    trainer = pl.Trainer(devices=[0], max_steps=200000, logger=wandb_logger,
                         val_check_interval=2000, limit_val_batches=5)
    trainer.fit(model, loader, loader)
    trainer.save_checkpoint('models/sae_0.5B_l1=1e-2_lr=2e-4_redpajama_8x.ckpt')
    print('stop')