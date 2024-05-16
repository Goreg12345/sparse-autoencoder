from dataclasses import asdict

import pytorch_lightning as pl
import torch
from datasets import load_dataset
from metrics.small_decoder_norm import SmallDecoderNormCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

import text_dataset
from SparseAutoencoder import SparseAutoencoder
from activation_buffer import Buffer, DiscBuffer
from metrics.L0Loss import L0LossCallback
from metrics.dead_neurons import DeadNeuronsCallback
from metrics.reconstruction_loss import ReconstructionLossCallback
from metrics.ultra_low_density_neurons import UltraLowDensityNeuronsCallback
from text_dataset import TextDataset
from training.DeadFeatureResampler import DeadFeatureResampler
from training.config import SAEConfig


def train(config: SAEConfig):

    dataset = load_dataset(config.dataset_name, split="train")
    if "TinyStories" in str(dataset) or "pile" in str(dataset):
        dataset = dataset["train"]
    dataset = dataset.shuffle()

    llm = HookedTransformer.from_pretrained(
        model_name=config.language_model,
        device="cpu",  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    token_dataset = TextDataset(
        dataset,
        llm.to_tokens,
        config.extraction_batch_size,
        drop_last_batch=False,
        seq_len=config.seq_len,
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

    if config.use_disc_dataset:
        config.store_path = f'{config.store_path}/{config.actv_name.split("_")[-1]}-l{config.layer}h{config.head}-{(config.store_size / 1e6):.0f}M.h5'
        buffer = DiscBuffer(config.store_path, accessor=config.store_accessor)
        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(buffer),
            config.batch_size,
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
        buffer = Buffer(llm.cuda(), text_dataset_loader, **asdict(config))
        loader = torch.utils.data.DataLoader(
            buffer, batch_size=None, shuffle=False, num_workers=0
        )

    # AUTOENCODER
    sae = SparseAutoencoder(config)

    reconstruction_loss_metric_zero = ReconstructionLossCallback(
        llm,
        sae,
        TextDataset(
            dataset,
            llm.to_tokens,
            config.reconstruction_loss_batch_size,
            drop_last_batch=False,
            seq_len=config.seq_len,
        ),
        config.actv_name,
        config.head,
        ablation_type="zero",
    )
    reconstruction_loss_metric_mean = ReconstructionLossCallback(
        llm,
        sae,
        TextDataset(
            dataset,
            llm.to_tokens,
            config.reconstruction_loss_batch_size,
            drop_last_batch=False,
            seq_len=config.seq_len,
        ),
        config.actv_name,
        config.head,
        ablation_type="mean",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models",  # Specify your checkpoint directory
        filename="{step}",  # Optional: customize your checkpoint filename
        every_n_train_steps=122070,  # Save a checkpoint every n training steps
        save_last=-1,  # Optional: set to -1 to save all checkpoints, adjust as needed
    )
    model = sae.create_trainer()
    # TRAINING

    # Instantiate the metrics callbacks
    dead_neurons_callback = DeadNeuronsCallback(n_features=config.d_hidden, return_neuron_indices=False)
    l0_loss_callback = L0LossCallback(log_interval=2000, calc_on="validation")
    small_decoder_norm_callback = SmallDecoderNormCallback(n_features=config.d_hidden)
    ultra_low_density_neurons_callback = UltraLowDensityNeuronsCallback(n_features=config.d_hidden)

    # Training
    wandb_logger = pl.loggers.WandbLogger(
        project="serimats", config=asdict(config), name=config.wandb_name, log_model="all"
    )
    trainer = pl.Trainer(
        devices=[0],
        max_steps=config.train_steps,
        logger=wandb_logger,
        val_check_interval=5000,
        limit_val_batches=10,
        limit_test_batches=25,
        callbacks=[checkpoint_callback, dead_neurons_callback, l0_loss_callback,
                   reconstruction_loss_metric_zero, reconstruction_loss_metric_mean,
                   small_decoder_norm_callback, ultra_low_density_neurons_callback],
    )
    trainer.fit(model, loader, loader)

    if hasattr(buffer, "close"):
        buffer.close()
