import torch
from torch.utils.data import IterableDataset, Dataset
import h5py


class Buffer(IterableDataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll automatically run the model to generate more when it gets halfway empty.
    """

    def __init__(
        self,
        llm,
        dataset,
        buffer_size,
        actv_size,
        seq_len,
        extraction_batch_size,
        actv_name,
        layer,
        batch_size,
        head=None,
        **kwargs
    ):
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
        print("setting up now")
        self.refresh()

    @torch.no_grad()
    def add_batch_to_buffer(self, batch):
        batch[:, 0] = self.llm.tokenizer.bos_token_id
        names_filter = lambda name_: self.actv_name in name_
        batch = batch.to(self.llm.W_E.device)
        _, cache = self.llm.run_with_cache(
            batch, stop_at_layer=self.layer + 1, names_filter=names_filter
        )
        acts = cache[self.actv_name]
        if len(acts.shape) == 4:  # batch seq head dim
            if self.head == "concat":
                acts = acts.reshape(-1, acts.shape[-1] * acts.shape[-2])
            else:
                acts = acts[..., self.head, :]
                acts = acts.reshape(-1, acts.shape[-1])
        else:
            acts = acts.view(-1, self.actv_size)  # view faster?

        # if the buffer is close to full, we might not be able to fit the whole batch in
        end = (
            self.pointer + acts.shape[0]
            if self.pointer + acts.shape[0] < self.buffer.shape[0]
            else self.buffer.shape[0]
        )
        # self.buffer[self.pointer:end] = acts[:end - self.pointer]
        # copy asynchronously to the cpu buffer because these are multiple GB of activations
        self.buffer[self.pointer : end].copy_(
            acts[: end - self.pointer], non_blocking=True
        )
        self.pointer = end
        print("new batch added to buffer")

    @torch.no_grad()
    def refresh(self):
        self.buffer = torch.empty(
            (int(self.buffer_size), self.actv_size),
            requires_grad=False,
            device="cpu",
            dtype=torch.float32,
        ).pin_memory()
        # TODO: if last batch is too small, need to add it to the buffer and then resize the buffer
        self.pointer = 0
        while True:
            batch = next(self.dataset)
            # if batch is smaller than extraction batch size, it's the last batch and we need to resize the buffer
            if batch.shape[0] < self.extraction_batch_size:
                self.add_batch_to_buffer(batch)
                self.buffer = self.buffer[: self.pointer]
                break
            # if buffer is full, stop
            if self.pointer >= self.buffer.shape[0]:
                break
            self.add_batch_to_buffer(batch)

        self.pointer = 0
        torch.cuda.synchronize()  # wait for the cpu buffer to be filled via asynchronous copying from gpu memory
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).to("cpu")]

    def __iter__(self):
        return self

    @torch.no_grad()
    def __next__(self):
        out = self.buffer[self.pointer : self.pointer + self.batch_size]
        self.pointer += self.batch_size
        if self.pointer > self.buffer.shape[0] - self.batch_size:
            # print("Refreshing the buffer!")
            self.refresh()
        return out


# class DiscBuffer(IterableDataset):
#    """
#    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It'll
#    automatically run the model to generate more when it gets halfway empty.
#    """
#
#    def __init__(self, h5_path, batch_size, accessor='tensor', **kwargs):
#        self.batch_size = batch_size
#        self.h5_path = h5_path
#        self.accessor = accessor
#        self.h5 = h5py.File(self.h5_path, 'r')
#        self.pointer = 0
#        self.buffer = self.h5[self.accessor]
#
#    def close(self, exc_type, exc_value, traceback):
#        if self.h5:
#            self.h5.close()
#        if exc_type:
#            print(exc_type, exc_value, traceback)
#
#    def __iter__(self):
#        return self
#
#    @torch.no_grad()
#    def __next__(self):
#        out = self.buffer[self.pointer:self.pointer + self.batch_size]
#        out = torch.tensor(out, dtype=torch.float32)
#        self.pointer += self.batch_size
#        if self.pointer > self.buffer.shape[0] - self.batch_size:
#            raise StopIteration
#        return out


class DiscBuffer(Dataset):
    """
    This defines a data buffer, to store a bunch of MLP acts that can be used to train the autoencoder. It provides
    random access to the elements stored in an HDF5 file.
    """

    def __init__(self, h5_path, accessor="tensor", **kwargs):
        self.h5_path = h5_path
        self.accessor = accessor
        # Open the HDF5 file in read mode for faster access
        self.h5 = h5py.File(self.h5_path, "r", **kwargs)
        self.buffer = self.h5[self.accessor]

    def __len__(self):
        # Return the total number of items in the dataset
        return self.buffer.shape[0]

    @torch.no_grad()
    def __getitem__(self, idx):
        # Fetch the data item at the specified index
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Efficiently fetch the data from the HDF5 file
        data = torch.tensor(self.buffer[idx], dtype=torch.float32)
        return data

    def close(self):
        # Close the HDF5 file when done
        if self.h5:
            self.h5.close()
