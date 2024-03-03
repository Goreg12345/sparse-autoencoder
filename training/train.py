import argparse
import os

# init torch only after setting the GPU
import sys


def linear_growth_scheduler(batch_idx, l1_coefficient):
    return l1_coefficient
    # base_value = l1_coefficient.clone()
    # final_step = 5000
    # start_multiplier = 0.01
    # if batch_idx >= final_step:
    #     return base_value
    # start_value = base_value * start_multiplier
    # # Linear interpolation from start_value to base_value
    # return start_value + (base_value - start_value) * (batch_idx / final_step)


def train(config):
    import torch
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
        config[
            "store_path"
        ] = f'{config["store_path"]}/{config["actv_name"].split("_")[-1]}-l{config["layer"]}h{config["head"]}-{(config["store_size"] / 1e6):.0f}M.h5'
        buffer = DiscBuffer(config["store_path"], accessor=config["store_accessor"])
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
        every_n_train_steps=122070,  # Save a checkpoint every n training steps
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
        start_lr_decay=config["start_lr_decay"],
        end_lr_decay=config["end_lr_decay"],
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


if __name__ == "__main__":
    # Default configuration
    cfg = {
        "actv_size": 64,
        "buffer_size": 1e7,
        "extraction_batch_size": 100,
        "actv_name": "blocks.9.attn.hook_z",
        "layer": 9,
        "seq_len": 512,
        "dataset_name": "Skylion007/openwebtext",
        "language_model": "gpt2-small",
        "use_disc_dataset": False,
        "store_path": "/var/local/glang/activations",
        "store_accessor": "tensor",
        "store_size": int(1e9),
        "train_steps": 122071,  # 250 M
        "batch_size": 2048,
        "d_hidden": 64 * 128,
        "l1_coefficient": 0.006,
        "standardize_activations": True,
        "adjust_for_dict_size": False,
        "init_geometric_median": False,
        "allow_lower_decoder_norm": False,
        "disable_decoder_bias": False,
        "lr": 1e-3,
        "beta1": 0.9,
        "beta2": 0.999,
        "start_lr_decay": 0,
        "end_lr_decay": 0,
        "resampling_steps": [],  # , 100000, 150000, 200000],
        "n_resampling_watch_steps": 5000,
        "lr_warmup_steps": 6000,
        "head": 9,
        "l1_exponent": 1,
        "reconstruction_loss_batch_size": 16,
        "n_resampler_samples": 819200,
        "wandb_name": "",
        "ckpt_name": "",
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
        "--buffer_size",
        type=float,
        help="Buffer size, should be big, e.g. 1e7 to remove correlation of "
        "tokens of the same context",
    )
    parser.add_argument(
        "--extraction_batch_size", type=int, help="Extraction batch size"
    )
    parser.add_argument("--actv_name", type=str, help="Activation function name")
    parser.add_argument("--layer", type=int, help="Transformer layer to hook")
    parser.add_argument("--seq_len", type=int, help="Sequence length")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--language_model", type=str, help="Pretrained language model")
    parser.add_argument(
        "--use_disc_dataset",
        type=bool,
        help="Whether to use the disk dataset or freshly fetch "
        "activations by running text through the model",
    )
    parser.add_argument(
        "--store_path",
        type=str,
        help="Path to access the h5 file of the stored activations",
    )
    parser.add_argument(
        "--store_accessor", type=str, help="Accessor to use for the store"
    )
    parser.add_argument("--train_steps", type=int, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--d_hidden", type=int, help="Hidden layer dimension")
    parser.add_argument(
        "--l1_coefficient", type=float, help="L1 regularization coefficient"
    )
    parser.add_argument(
        "--standardize_activations",
        type=bool,
        help="Whether to standardize activations",
    )
    parser.add_argument(
        "--adjust_for_dict_size",
        type=bool,
        help="Whether to adjust for dictionary size. If true, it standardizes activations to have norm sqrt(d_hidden) and will sum mse across the dense dimension",
    )
    parser.add_argument(
        "--init_geometric_median",
        type=bool,
        help="Whether to initialize using geometric median",
    )
    parser.add_argument(
        "--allow_lower_decoder_norm",
        type=bool,
        help="Whether to allow lower decoder norm",
    )
    parser.add_argument(
        "--disable_decoder_bias", type=bool, help="Whether to disable decoder bias"
    )
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--beta1", type=float, help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, help="Beta2 for Adam optimizer")
    parser.add_argument(
        "--start_lr_decay",
        type=int,
        help="Start learning rate decay step. Will linearly reduce "
        "learning rate from start to end over the decay steps. Set "
        "to 0 to disable.",
    )
    parser.add_argument("--end_lr_decay", type=int, help="End learning rate decay step")
    parser.add_argument(
        "--resampling_steps",
        type=int,
        nargs="+",
        help="Steps at which to resample dead features",
    )
    parser.add_argument(
        "--n_resampling_watch_steps",
        type=int,
        help="Number of steps to watch for resampling",
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, help="Number of warmup steps for learning rate"
    )
    parser.add_argument("--head", type=int, help="Head number for multi-head attention")
    parser.add_argument(
        "--l1_exponent",
        type=float,
        help="Exponent for L1 regularization, use 1 for L1 or 0.5 for l0.5",
    )
    parser.add_argument(
        "--reconstruction_loss_batch_size",
        type=int,
        help="Batch size for reconstruction loss calculation",
    )
    parser.add_argument(
        "--n_resampler_samples", type=int, help="Number of samples for resampler"
    )
    parser.add_argument(
        "--wandb_name", type=str, help="Weights & Biases experiment name"
    )
    parser.add_argument(
        "--ckpt_name", type=str, help="Checkpoint name for saving/loading model"
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

    cfg[
        "wandb_name"
    ] = f"{cfg['actv_name'].split('_')[-1]}-{cfg['wandb_name']}-l{cfg['layer']}h{cfg['head']}-l1_{cfg['l1_coefficient']}-{(cfg['train_steps'] * cfg['batch_size'] / 1e6):.0f}M"
    cfg[
        "ckpt_name"
    ] = f"{cfg['actv_name'].split('_')[-1]}-l{cfg['layer']}h{cfg['head']}-l1_{cfg['l1_coefficient']}-{(cfg['train_steps'] * cfg['batch_size'] / 1e6):.0f}M.ckpt"

    # Train the model
    train(cfg)
