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

from DeadFeatureResampler import DeadFeatureResampler
from HookedSparseAutoencoder import HookedSparseAutoencoder
from SparseAutoencoder import SparseAutoencoder
from metrics.reconstruction_loss import ReconstructionLoss
from neel_buffer import Buffer
from text_dataset import TextDataset


cfg = {
    # EXTRACTION
    'actv_size': 64, # 768,
    'buffer_size': 1e7,
    'extraction_batch_size': 130,
    'actv_name': 'blocks.10.attn.hook_z', # hook_mlp_out',  # 'blocks.8.hook_resid_post',
    'layer': 10,
    'seq_len': 512,
    'dataset_name': "Skylion007/openwebtext",  # togethercomputer/RedPajama-Data-1T-Sample roneneldan/TinyStories

    # AUTOENCODER
    'train_steps': 49000,
    'batch_size': 2048,
    'd_hidden': 64 * 64,
    'l1_coefficient': 2e-2,#1e-3,# 1e-2,  # 1.4e-2,
    'lr': 2e-4,
    'resampling_steps': [50000, 100000, 150000, 200000],
    'n_resampling_watch_steps': 5000,
    'lr_warmup_steps': 3000,
    'head': 0,

    # METRICS
    'reconstruction_loss_batch_size': 16,
    'n_resampler_samples': 819200,

    # LOGGING
    'wandb_name': 'z-l10h0-64x-100M-1.2e-2',
    'ckpt_name': 'z-l10h0-64x-100M-1_2e-2', # 1B_l1=1.4e-2_lr=2e-4_owt_8x.ckpt',
}

if __name__=='__main__':
    # LOAD DATA
    dataset = load_dataset(cfg['dataset_name'], split='train')
    if 'TinyStories' in str(dataset) or 'pile' in str(dataset):
        dataset = dataset['train']
    dataset = dataset.shuffle()

    llm = HookedTransformer.from_pretrained(
        model_name='gpt2-small',
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
    sae = HookedSparseAutoencoder(d_input=cfg['actv_size'], d_hidden=cfg['d_hidden'])
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

    resampler = DeadFeatureResampler(sae, loader, cfg['n_resampler_samples'], cfg['actv_size'], cfg['d_hidden'])


    # TODO: currently off (set start multiplier)
    def linear_growth_scheduler(batch_idx):
        base_value = cfg['l1_coefficient']
        final_step = 5000
        start_multiplier = 0.01
        if batch_idx >= final_step:
            return base_value
        #elif any([(batch_idx - step) > 0 and (batch_idx - step < final_step) for step in cfg['resampling_steps']]):
        #    batch_idx = batch_idx % 5000
        start_value = base_value * start_multiplier
        # Linear interpolation from start_value to base_value
        return start_value + (base_value - start_value) * (batch_idx / final_step)

    # hf_run = 'concat-z-gelu-21-l1-lr-sweep-3/gelu-2l_L1_Hcat_z_lr1.00e-03_l12.00e+00_ds16384_bs4096_dc1.00e-07_rie50000_nr4_v78'
    # model = SparseAutoencoder.load_theirs(hf_run, llm, TextDataset(dataset, llm.to_tokens,
    #                                                             cfg['reconstruction_loss_batch_size'],
    #                                                             drop_last_batch=False,
    #                                                             seq_len=cfg['seq_len']),
    #                                       cfg, resampler, linear_growth_scheduler)
    model = SparseAutoencoder(sae, cfg['resampling_steps'], cfg['n_resampling_watch_steps'],
                              cfg['l1_coefficient'], reconstruction_loss_metric_zero, reconstruction_loss_metric_mean,
                              resampler, cfg['lr'],
                              l1_scheduler=linear_growth_scheduler, lr_warmup_steps=cfg['lr_warmup_steps'],)

    # TRAINING
    wandb_logger = pl.loggers.WandbLogger(project='serimats', config=cfg, name=cfg['wandb_name'])
    trainer = pl.Trainer(devices=[0], max_steps=cfg['train_steps'], logger=wandb_logger,
                         val_check_interval=2000, limit_val_batches=5, limit_test_batches=5)
    trainer.fit(model, loader, loader)
    trainer.save_checkpoint(f'models/{cfg["ckpt_name"]}.ckpt')
