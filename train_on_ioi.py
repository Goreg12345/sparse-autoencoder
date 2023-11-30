import torch
import datasets
# %%
from datasets import load_dataset

dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
# %%
cfg = {
    'actv_size': 768,
    'buffer_batches': 100,
    'buffer_size': 1e7,
    'extraction_batch_size': 500,
    'actv_name': 'blocks.8.hook_resid_post',
    'layer': 8,
    'batch_size': 2048,
    'seq_len': 512
}
# %%
from transformer_lens import HookedTransformer

llm = HookedTransformer.from_pretrained(
    model_name='gpt2-small',
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device='cpu'  # will be moved to GPU later by lightning
)
llm.requires_grad_(False)

text_dataset = TextDataset(dataset, llm.to_tokens, cfg['extraction_batch_size'], drop_last_batch=False,
                           seq_len=cfg['seq_len'])
# don't increase num_workers > 1 because it's an IterableDataset and multiple dataloaders will yield the same data
text_dataset_loader = iter(DataLoader(text_dataset, batch_size=None, shuffle=False, num_workers=1, prefetch_factor=200))
next(text_dataset_loader)
buffer = Buffer(
    llm.cuda(),
    text_dataset_loader,
    **cfg
)
loader = torch.utils.data.DataLoader(buffer, batch_size=None, shuffle=False, num_workers=0)

d_hidden = cfg['actv_size'] * 8
d_hidden = int(2**16)
sae = HookedSparseAutoencoder(d_input=cfg['actv_size'], d_hidden=d_hidden)
reconstruction_loss_metric = ReconstructionLoss(llm, sae,
                                                TextDataset(dataset, llm.to_tokens, 16,
                                                            drop_last_batch=False,
                                                            seq_len=cfg['seq_len'])
                                                , cfg['actv_name'])

resampler = DeadFeatureResampler(sae, loader, 200000, cfg['actv_size'], d_hidden)

wandb_logger = pl.loggers.WandbLogger(project='serimats', config=cfg,
                                      name=f'sae_debug')
model = SparseAutoencoder(sae, 1e-2, reconstruction_loss_metric, resampler)
# load pretrained model
#model = SparseAutoencoder.load_from_checkpoint('models/sae_resampleDebug_22itcheck_l1=1e-2_lr=2e-4_redpajama_8x.ckpt')
trainer = pl.Trainer(devices=[0], max_steps=200000, logger=wandb_logger,
                     val_check_interval=2000, limit_val_batches=5)
trainer.fit(model, loader, loader)
trainer.save_checkpoint('models/sae_0.5B_l1=1e-2_lr=2e-4_redpajama_2e16.ckpt')
print('stop')