import sys, os

try:
    gpu_num = int(sys.argv[1])
except:
    gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

import torch
from SAETrainer import SAETrainer
import wandb
import text_dataset
from text_dataset import TextDataset
from torch.utils.data import DataLoader
from activation_buffer import Buffer
from metrics.reconstruction_loss import ReconstructionLoss
import pytorch_lightning as pl
from datasets import load_dataset
from transformer_lens import HookedTransformer


def sanity_check(ckpt_path, cfg, encoder):
    dataset = load_dataset(cfg["dataset_name"], split="train")
    if "TinyStories" in str(dataset) or "pile" in str(dataset):
        dataset = dataset["train"]
    dataset = dataset.shuffle()

    llm = HookedTransformer.from_pretrained(
        model_name="gpt2-small",
        # refactor_factored_attn_matrices=True,
        device="cpu",  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    # this yields batches of tokens with sequence length cfg['seq_len']
    token_dataset = TextDataset(
        dataset,
        llm.to_tokens,
        encoder.cfg["extraction_batch_size"],
        drop_last_batch=False,
        seq_len=encoder.cfg["seq_len"],
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

    # this yields batches of activations
    buffer_size = cfg["buffer_size"]
    cfg["buffer_size"] = 1e5  # make it small for the test to be fast
    buffer = Buffer(llm.cuda(), text_dataset_loader, **cfg)
    cfg["buffer_size"] = buffer_size
    loader = torch.utils.data.DataLoader(
        buffer, batch_size=None, shuffle=False, num_workers=0
    )

    reconstruction_loss_metric_zero = ReconstructionLoss(
        llm,
        encoder,
        TextDataset(
            dataset,
            llm.to_tokens,
            cfg["reconstruction_loss_batch_size"],
            drop_last_batch=False,
            seq_len=cfg["seq_len"],
        ),
        cfg["actv_name"],
        cfg["head"],
        ablation_type="zero",
    )
    reconstruction_loss_metric_mean = ReconstructionLoss(
        llm,
        encoder,
        TextDataset(
            dataset,
            llm.to_tokens,
            cfg["reconstruction_loss_batch_size"],
            drop_last_batch=False,
            seq_len=cfg["seq_len"],
        ),
        cfg["actv_name"],
        cfg["head"],
        ablation_type="mean",
    )

    encoder_trainer = SAETrainer.load_from_checkpoint(
        ckpt_path,
        cfg,
        reconstruction_loss_metric_mean=reconstruction_loss_metric_mean,
        reconstruction_loss_metric_zero=reconstruction_loss_metric_zero,
    )
    trainer = pl.Trainer(limit_test_batches=5)
    from IPython.display import display, HTML, clear_output

    clear_output()
    trainer.test(encoder_trainer, loader)


if __name__ == "__main__":
    api = wandb.Api()
    run = api.run("mega-alignment/sae/9jyap5cc")  # this is the q-l9h9 sae with low l0
    cfg = run.config
    artifact_dir = api.artifact("georglange/serimats/model-9jyap5cc:v1").download()

    ckpt_path = os.path.join(artifact_dir, "model.ckpt")
    encoder = SAETrainer.load_from_checkpoint(
        ckpt_path,
        cfg,
    ).sae

    sanity_check(ckpt_path, cfg, encoder)
