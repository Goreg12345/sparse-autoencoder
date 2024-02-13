from DeadFeatureResampler import DeadFeatureResampler

import torch
import einops
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

from SparseAutoencoder import SparseAutoencoder
from SAETrainer import SAETrainer
from metrics.reconstruction_loss import ReconstructionLoss
from text_dataset import TextDataset
import time

class Buffer(IterableDataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(self, llm, dataset, buffer_size, actv_size, seq_len, extraction_batch_size, actv_name, layer, batch_size, head=None, **kwargs ):
        self.buffer_size = buffer_size
        self.token_pointer = 0
        self.first = True
        self.llm = llm
        self.extraction_batch_size = extraction_batch_size
        self.actv_name = actv_name
        self.layer = layer
        self.actv_size = actv_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        if head is not None:
            self.head = head
        self.dataset = dataset
        print('setting up now')
        self.refresh()

    def add_batch_to_buffer(self, batch):
        batch[:, 0] = self.llm.tokenizer.bos_token_id
        names_filter = lambda name_: self.actv_name in name_
        batch = batch.to(self.llm.W_E.device)
        _, cache = self.llm.run_with_cache(batch, stop_at_layer=self.layer + 1, names_filter=names_filter)
        acts = cache[self.actv_name]
        if len(acts.shape) == 4:  # batch seq head dim
            if self.head == 'concat':
                acts = acts.reshape(-1, acts.shape[-1] * acts.shape[-2])
            else:
                acts = acts[..., self.head, :]
                acts = acts.reshape(-1, acts.shape[-1])
        else:
            acts = acts.view(-1, self.actv_size)  # view faster?

        # if the buffer is close to full, we might not be able to fit the whole batch in
        end = self.pointer + acts.shape[0] if self.pointer + acts.shape[0] < self.buffer.shape[0] else self.buffer.shape[0]
        # self.buffer[self.pointer:end] = acts[:end - self.pointer]
        # copy asynchronously to the cpu buffer because these are multiple GB of activations
        self.buffer[self.pointer:end].copy_(acts[:end - self.pointer], non_blocking=True)
        self.pointer = end
        print('new batch added to buffer')

    @torch.no_grad()
    def refresh(self):
        self.buffer = torch.empty((int(self.buffer_size), self.actv_size), requires_grad=False, device='cpu', dtype=torch.float32).pin_memory()
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
        torch.cuda.synchronize()  # wait for the cpu buffer to be filled via asynchronous copying from gpu memory
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
