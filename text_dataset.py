from typing import Callable

import torch
from torch.utils.data import IterableDataset


class TextDataset(IterableDataset):
    def __init__(self, hf_dataset, to_tokens: Callable, batch_size, drop_last_batch=False, hf_text_accessor='text',
                 seq_len=128,):
        """
        Takes a huggingface dataset and returns batches of tokens
        :param hf_dataset: huggingface dataset that contains the text
        :param to_tokens: function that converts text to tokens, e.g. the tokenizer function or HookedTransformer.to_tokens()
        :param batch_size: batch size
        :param drop_last_batch: if True, the last batch will be dropped if it's smaller than batch_size
        :param hf_text_accessor: str, key to access the text in the hf_dataset
        :param seq_len: int, sequence length per sample in the batch
        returns batches of shape (batch_size, seq_len), filled with tokens
        """
        self.hf_dataset = hf_dataset
        self.to_tokens = to_tokens
        self.token_pointer = 0
        self.drop_last_batch = drop_last_batch
        self.hf_text_accessor = hf_text_accessor
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.tokens = []

        dataset_len = len(self.hf_dataset)
        # create a permutation of the dataset indices to shuffle it
        self.permutation = torch.randperm(dataset_len).to('cpu')

    def __iter__(self):
        return self

    def __next__(self):
        batch = torch.empty((self.batch_size, self.seq_len), dtype=torch.long).to('cuda')

        while True:
            # if dataset is exhausted, stop
            if self.token_pointer == len(self.hf_dataset):
                if self.drop_last_batch:
                    raise StopIteration
                else:
                    return batch[:self.batch_pointer]

            # get a new sample and add it to the batch
            # self.tokens += self.to_tokens(self.hf_dataset[self.token_pointer][self.hf_text_accessor], prepend_bos=False)[0]
            self.tokens += self.to_tokens(self.hf_dataset[self.permutation[self.token_pointer].view(-1)][self.hf_text_accessor],
                                          prepend_bos=False)[0]
            self.token_pointer += 1

            # fill the batch row by row with tokens until we need to sample more or the batch is full
            while len(self.tokens) > self.seq_len:
                batch[self.batch_pointer] = torch.tensor(self.tokens[:self.seq_len])
                self.tokens = self.tokens[self.seq_len:]
                self.batch_pointer += 1
                if self.batch_pointer == self.batch_size:
                    break

            # if batch is full, return it
            if self.batch_pointer == self.batch_size:
                self.batch_pointer = 0
                return batch
