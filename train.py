import os
import text_dataset
import sys

# get first argument from command line
gpu_num = int(sys.argv[1])

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
# execute system command
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from DeadFeatureResampler import DeadFeatureResampler
from SparseAutoencoder import SparseAutoencoder
from SAETrainer import SAETrainer
from metrics.reconstruction_loss import ReconstructionLoss
from activation_buffer import Buffer
from text_dataset import TextDataset


cfg = {
    # EXTRACTION
    'actv_size': 64, # 768,
    'buffer_size': 1e7,
    'extraction_batch_size': 100,
    'actv_name': 'blocks.10.attn.hook_z', # hook_mlp_out',  # 'blocks.8.hook_resid_post',
    'layer': 10,
    'seq_len': 512,
    'dataset_name': "Skylion007/openwebtext",  # togethercomputer/RedPajama-Data-1T-Sample roneneldan/TinyStories
    'language_model': 'gpt2-small',

    # AUTOENCODER
    'train_steps': 400000,
    'batch_size': 2048,
    'd_hidden': 64 * 128,
    'l1_coefficient': 0.06,# 7e-1,#1e-3,# 1e-2,  # 1.4e-2,
    'standardize_activations': True,
    'init_geometric_median': False,
    'lr': 1e-3,
    'resampling_steps': [50000, 100000, 150000, 200000],
    'n_resampling_watch_steps': 5000,
    'lr_warmup_steps': 3000,
    'head': 0,

    # METRICS
    'reconstruction_loss_batch_size': 16,
    'n_resampler_samples': 819200,

    # LOGGING
    'wandb_name': '',
    'ckpt_name': '', #'q-l9h9-40M-l1_5e-2.ckpt', # 1B_l1=1.4e-2_lr=2e-4_owt_8x.ckpt',
}

cfg['wandb_name'] = f"{cfg['actv_name'].split('_')[-1]}-{cfg['wandb_name']}-l{cfg['layer']}h{cfg['head']}-l1_{cfg['l1_coefficient']}-{cfg['train_steps'] * cfg['batch_size'] / 1e6}M"
cfg['ckpt_name'] = f"{cfg['actv_name'].split('_')[-1]}-l{cfg['layer']}h{cfg['head']}-l1_{cfg['l1_coefficient']}-{cfg['train_steps'] * cfg['batch_size'] / 1e6}M.ckpt"


def linear_growth_scheduler(batch_idx):
    base_value = cfg['l1_coefficient']
    final_step = 5000
    start_multiplier = 0.01
    if batch_idx >= final_step:
        return base_value
    start_value = base_value * start_multiplier
    # Linear interpolation from start_value to base_value
    return start_value + (base_value - start_value) * (batch_idx / final_step)

if __name__=='__main__':
    # LOAD DATA
    dataset = load_dataset(cfg['dataset_name'], split='train')
    if 'TinyStories' in str(dataset) or 'pile' in str(dataset):
        dataset = dataset['train']
    dataset = dataset.shuffle()

    llm = HookedTransformer.from_pretrained(
        model_name=cfg['language_model'],
        # refactor_factored_attn_matrices=True,
        device='cpu'  # will be moved to GPU later by lightning
    )
    llm.requires_grad_(False)

    token_dataset = TextDataset(dataset, llm.to_tokens, cfg['extraction_batch_size'], drop_last_batch=False,
                                seq_len=cfg['seq_len'])

    text_dataset_loader = iter(DataLoader(token_dataset, batch_size=None, shuffle=False, num_workers=5,
                                          prefetch_factor=5, worker_init_fn=text_dataset.worker_init_fn))
    buffer = Buffer(
        llm.cuda(),
        text_dataset_loader,
        **cfg
    )
    loader = torch.utils.data.DataLoader(buffer, batch_size=None, shuffle=False, num_workers=0)

    # AUTOENCODER
    sae = SparseAutoencoder(d_input=cfg['actv_size'], d_hidden=cfg['d_hidden'], cfg=cfg)
    if cfg['standardize_activations']:
        sae.init_activation_standardization(buffer.buffer)
    if cfg['init_geometric_median']:
        sae.init_geometric_median(buffer.buffer)
    reconstruction_loss_metric_zero = ReconstructionLoss(llm, sae,
                                                    TextDataset(dataset, llm.to_tokens,
                                                                cfg['reconstruction_loss_batch_size'],
                                                                drop_last_batch=False,
                                                                seq_len=cfg['seq_len'])
                                                    , cfg['actv_name'],
                                                    cfg['head'], ablation_type='zero')
    reconstruction_loss_metric_mean = ReconstructionLoss(llm, sae,
                                                    TextDataset(dataset, llm.to_tokens,
                                                                cfg['reconstruction_loss_batch_size'],
                                                                drop_last_batch=False,
                                                                seq_len=cfg['seq_len'])
                                                    , cfg['actv_name'],
                                                    cfg['head'], ablation_type='mean')

    resampler = DeadFeatureResampler(loader, cfg['n_resampler_samples'], cfg['actv_size'], cfg['d_hidden'])
    checkpoint_callback = ModelCheckpoint(
        dirpath='models',  # Specify your checkpoint directory
        filename='{step}',  # Optional: customize your checkpoint filename
        every_n_train_steps=19999,  # Save a checkpoint every n training steps
        save_last=True,  # Optional: set to -1 to save all checkpoints, adjust as needed
    )
    model = SAETrainer(sae, cfg['resampling_steps'], cfg['n_resampling_watch_steps'],
                       cfg['l1_coefficient'], reconstruction_loss_metric_zero, reconstruction_loss_metric_mean,
                       resampler, cfg['lr'],
                       l1_scheduler=linear_growth_scheduler, lr_warmup_steps=cfg['lr_warmup_steps'], )

    # TRAINING

    wandb_logger = pl.loggers.WandbLogger(project='serimats', config=cfg, name=cfg['wandb_name'], log_model=True)
    trainer = pl.Trainer(devices=[0], max_steps=cfg['train_steps'], logger=wandb_logger,
                         val_check_interval=2000, limit_val_batches=5, limit_test_batches=5,
                        callbacks=[checkpoint_callback]
                         )
    trainer.fit(model, loader, loader)
