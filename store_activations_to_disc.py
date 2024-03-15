import argparse
import os

# Default configuration
cfg = {
    "actv_size": 64,
    "store_size": int(2.5e8),
    "buffer_size": 1e7,
    "batch_size": int(1e6),
    "extraction_batch_size": 120,
    "actv_name": "blocks.9.attn.hook_z",
    "layer": 9,
    "seq_len": 512,
    "dataset_name": "Skylion007/openwebtext",
    "language_model": "gpt2-small",
    "head": 6,
    "store_path": "/var/local/glang/activations",
}

parser = argparse.ArgumentParser(
    description="Train an SAE with configurable parameters"
)

# Add arguments for configuration parameters
parser.add_argument("--gpu_num", type=int, help="GPU number to use")
parser.add_argument(
    "--actv_size",
    type=int,
    help="Activation size, i.e. number of neurons in the layer to hook",
)
parser.add_argument(
    "--store_size",
    type=float,
    help="Number of activations in store, should be big, e.g. 1e9 to have enough activations for training",
)
parser.add_argument("--extraction_batch_size", type=int, help="Extraction batch size")
parser.add_argument("--actv_name", type=str, help="Activation function name")
parser.add_argument("--batch_size", type=int, help="Batch size for training")
parser.add_argument("--layer", type=int, help="Transformer layer to hook")
parser.add_argument("--seq_len", type=int, help="Sequence length")
parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
parser.add_argument("--language_model", type=str, help="Pretrained language model")
parser.add_argument("--head", type=int, help="Head number for multi-head attention")
parser.add_argument(
    "--buffer_size",
    type=float,
    help="Buffer size, should be big, e.g. 1e7 to remove correlation of tokens of the same context",
)

# Parse arguments
args = parser.parse_args()

# Set GPU number from command line argument, if provided
if args.gpu_num is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)

# Update cfg with any command-line arguments provided
for key, value in vars(args).items():
    if value is not None:
        cfg[key] = value

# init torch only after setting the GPU
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import h5py
from tqdm import tqdm

from text_dataset import TextDataset
import text_dataset
from activation_buffer import Buffer


if __name__ == "__main__":
    # LOAD DATA
    dataset = load_dataset(cfg["dataset_name"], split="train")
    if "TinyStories" in str(dataset) or "pile" in str(dataset):
        dataset = dataset["train"]
    # dataset = dataset.shuffle()  # we don't need to shuffle this because we're going to shuffle by writing to the storage in a permutation

    llm = HookedTransformer.from_pretrained(
        model_name=cfg["language_model"],
        # refactor_factored_attn_matrices=True,
        device="cpu",
    )
    llm.requires_grad_(False)

    token_dataset = TextDataset(
        dataset,
        llm.to_tokens,
        cfg["extraction_batch_size"],
        drop_last_batch=False,
        seq_len=cfg["seq_len"],
    )

    text_dataset_loader = iter(
        DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=5,
            prefetch_factor=5,
            worker_init_fn=text_dataset.worker_init_fn,
        )
    )

    # device = "cpu" if cfg["gpu_num"] is None else f'cuda:{cfg["gpu_num"]}'
    llm.cuda()

    filename = f'{cfg["actv_name"].split("_")[-1]}-l{cfg["layer"]}h{cfg["head"]}-{(cfg["store_size"] / 1e6):.0f}M.h5'

    buffer = Buffer(llm.cuda(), text_dataset_loader, **cfg)
    loader = torch.utils.data.DataLoader(
        buffer, batch_size=None, shuffle=False, num_workers=0
    )

    with h5py.File(os.path.join(cfg["store_path"], filename), "w") as f:
        # Create a dataset within the file
        dset = f.create_dataset(
            "tensor", (cfg["store_size"], cfg["actv_size"]), dtype="float32"
        )

        h5_pointer = 0
        for batch in tqdm(loader, total=int(cfg["store_size"] / cfg["batch_size"])):
            if h5_pointer + cfg["batch_size"] > cfg["store_size"]:
                dset[h5_pointer:] = (
                    batch[: cfg["store_size"] - h5_pointer].cpu().numpy()
                )
                break
            else:
                dset[h5_pointer : h5_pointer + cfg["batch_size"]] = batch.cpu().numpy()
                h5_pointer += cfg["batch_size"]
