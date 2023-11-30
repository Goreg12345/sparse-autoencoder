import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import pytorch_lightning as pl

from DeadFeatureResampler import DeadFeatureResampler
from HookedSparseAutoencoder import HookedSparseAutoencoder
from SparseAutoencoder import SparseAutoencoder
from metrics.reconstruction_loss import ReconstructionLoss
from neel_buffer import Buffer
from text_dataset import TextDataset

cfg = {
    # EXTRACTION
    'actv_size': 768,
    'buffer_size': 1e5,
    'extraction_batch_size': 500,
    'actv_name': 'blocks.8.mlp.hook_post',  # 'blocks.8.hook_resid_post',
    'layer': 8,
    'seq_len': 512.,
    'dataset_name': "monology/pile-uncopyrighted",  # togethercomputer/RedPajama-Data-1T-Sample roneneldan/TinyStories

    # AUTOENCODER
    'train_steps': 500000,
    'batch_size': 4096,
    'd_hidden': 768 * 8,
    'l1_coefficient': 1.4e-2,
    'lr': 2e-4,
    'resampling_steps': [50000, 100000, 150000, 200000],
    'n_resampling_watch_steps': 10000,

    # METRICS
    'reconstruction_loss_batch_size': 16,
    'n_resampler_samples': 819200,

    # LOGGING
    'wandb_name': 'sae_pile',
    'ckpt_name': 'sae_pile_2B_l1=1.4e-2_lr=2e-4_pile_8x.ckpt',
}

if __name__=='__main__':
    # LOAD DATA
    dataset = load_dataset(cfg['dataset_name'], split="train")
    if 'TinyStories' in str(dataset) or 'pile' in str(dataset):
        dataset = dataset['train']

    llm = HookedTransformer.from_pretrained(
        model_name='gpt2-small',
        refactor_factored_attn_matrices=True,
        device='cpu'  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    text_dataset = TextDataset(dataset, llm.to_tokens, cfg['extraction_batch_size'], drop_last_batch=False,
                               seq_len=cfg['seq_len'])
    # don't increase num_workers > 1 because it's an IterableDataset and multiple dataloaders will yield the same data
    text_dataset_loader = iter(DataLoader(text_dataset, batch_size=None, shuffle=False, num_workers=1,
                                          prefetch_factor=200))
    buffer = Buffer(
        llm.cuda(),
        text_dataset_loader,
        **cfg
    )
    loader = torch.utils.data.DataLoader(buffer, batch_size=None, shuffle=False, num_workers=0)

    # AUTOENCODER
    sae = HookedSparseAutoencoder(d_input=cfg['actv_size'], d_hidden=cfg['d_hidden'])
    reconstruction_loss_metric = ReconstructionLoss(llm, sae,
                                                    TextDataset(dataset, llm.to_tokens,
                                                                cfg['reconstruction_loss_batch_size'],
                                                                drop_last_batch=False,
                                                                seq_len=cfg['seq_len'])
                                                    , cfg['actv_name'])

    resampler = DeadFeatureResampler(sae, loader, cfg['n_resampler_samples'], cfg['actv_size'], cfg['d_hidden'])
    model = SparseAutoencoder(sae, cfg['resampling_steps'], cfg['n_resampling_watch_steps'],
                              cfg['l1_coefficient'], reconstruction_loss_metric, resampler, cfg['lr'])

    # TRAINING
    wandb_logger = pl.loggers.WandbLogger(project='serimats', config=cfg, name=cfg['wandb_name'])
    trainer = pl.Trainer(devices=[0], max_steps=cfg['train_steps'], logger=wandb_logger,
                         val_check_interval=2000, limit_val_batches=5)
    trainer.fit(model, loader, loader)
    trainer.save_checkpoint(f'models/{cfg["ckpt_name"]}.ckpt')
