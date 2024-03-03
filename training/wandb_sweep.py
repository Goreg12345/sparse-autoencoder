import os

import torch
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from transformer_lens import HookedTransformer

from DeadFeatureResampler import DeadFeatureResampler
from SparseAutoencoder import SparseAutoencoder
from SAETrainer import SAETrainer
from metrics.reconstruction_loss import ReconstructionLoss
from activation_buffer import Buffer, DiscBuffer
from text_dataset import TextDataset
import text_dataset


def linear_growth_scheduler(batch_idx, l1_coefficient):
    return l1_coefficient
    # base_value = l1_coefficient
    # final_step = 5000
    # start_multiplier = 0.01
    # if batch_idx >= final_step:
    #     return base_value
    # start_value = base_value * start_multiplier
    # # Linear interpolation from start_value to base_value
    # return start_value + (base_value - start_value) * (batch_idx / final_step)


def train(config):
    dataset = load_dataset(config["dataset_name"], split="train")
    if "TinyStories" in str(dataset) or "pile" in str(dataset):
        dataset = dataset["train"]
    dataset = dataset.shuffle()

    llm = HookedTransformer.from_pretrained(
        model_name=config["language_model"],
        # refactor_factored_attn_matrices=True,
        device="cpu",  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    token_dataset = TextDataset(
        dataset,
        llm.to_tokens,
        config["extraction_batch_size"],
        drop_last_batch=False,
        seq_len=config["seq_len"],
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

    if config["use_disc_dataset"]:
        buffer_filename = os.path.join(
            config["store_path"],
            f'{config["actv_name"].split("_")[-1]}-l{config["layer"]}h{config["head"]}-{(config["store_size"] / 1e6):.0f}M.h5',
        )
        buffer = DiscBuffer(buffer_filename, accessor=config["store_accessor"])
        # loader = torch.utils.data.DataLoader(buffer, batch_size=cfg['batch_size'], shuffle=False, num_workers=1, prefetch_factor=10)
        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(buffer),
            config["batch_size"],
            drop_last=False,
        )
        loader = torch.utils.data.DataLoader(
            buffer,
            sampler=batch_sampler,
            num_workers=1,
            prefetch_factor=10,
            batch_size=None,
        )
        llm.cuda()
    else:
        buffer = Buffer(llm.cuda(), text_dataset_loader, **config)
        loader = torch.utils.data.DataLoader(
            buffer, batch_size=None, shuffle=False, num_workers=0
        )

    # AUTOENCODER
    sae = SparseAutoencoder(
        d_input=config["actv_size"], d_hidden=config["d_hidden"], cfg=config
    )
    if config["init_geometric_median"]:
        sae.init_geometric_median(torch.tensor(buffer.buffer[:10000000]))
    if config["standardize_activations"]:
        sae.init_activation_standardization(
            torch.tensor(buffer.buffer[:10000000])
        )  # use a subset in case buffer is very large
    reconstruction_loss_metric_zero = ReconstructionLoss(
        llm,
        sae,
        TextDataset(
            dataset,
            llm.to_tokens,
            config["reconstruction_loss_batch_size"],
            drop_last_batch=False,
            seq_len=config["seq_len"],
        ),
        config["actv_name"],
        config["head"],
        ablation_type="zero",
    )
    reconstruction_loss_metric_mean = ReconstructionLoss(
        llm,
        sae,
        TextDataset(
            dataset,
            llm.to_tokens,
            config["reconstruction_loss_batch_size"],
            drop_last_batch=False,
            seq_len=config["seq_len"],
        ),
        config["actv_name"],
        config["head"],
        ablation_type="mean",
    )

    resampler = DeadFeatureResampler(
        loader, config["n_resampler_samples"], config["actv_size"], config["d_hidden"]
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="models",  # Specify your checkpoint directory
        filename="{step}",  # Optional: customize your checkpoint filename
        every_n_train_steps=100000,  # Save a checkpoint every n training steps
        save_last=-1,  # Optional: set to -1 to save all checkpoints, adjust as needed
    )
    model = SAETrainer(
        sae,
        config["resampling_steps"],
        config["n_resampling_watch_steps"],
        config["l1_coefficient"],
        reconstruction_loss_metric_zero,
        reconstruction_loss_metric_mean,
        resampler,
        config["lr"],
        l1_scheduler=linear_growth_scheduler,
        lr_warmup_steps=config["lr_warmup_steps"],
        beta_1=config["beta1"],
        beta_2=config["beta2"],
    )

    # TRAINING

    wandb_logger = pl.loggers.WandbLogger(
        project="serimats", config=config, name=config["wandb_name"], log_model="all"
    )
    trainer = pl.Trainer(
        devices=[0],
        max_steps=config["train_steps"],
        logger=wandb_logger,
        val_check_interval=5000,
        limit_val_batches=10,
        limit_test_batches=25,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, loader, loader)

    if hasattr(buffer, "close"):
        buffer.close()


print("Starting sweep")
wandb_run = wandb.init(project="serimats")
print("Initialized wandb")
print("Config:", wandb.config)
config = wandb.config
train(config)
