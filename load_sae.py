import re
import sys, os

from SparseAutoencoder import SparseAutoencoder

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
    from IPython.display import clear_output

    clear_output()
    trainer.test(encoder_trainer, loader)


def is_available(node, project_name="sae-all-ioi-heads-low-l0", entity="mega-alignment"):
    api = wandb.Api()

    component_name, layer, head = node.component_name, node.layer, node.head
    runs = api.runs(path=f"{entity}/{project_name}")
    for run in runs:
        run_component = run.name.split('--')[0]
        run_layer = int(run.name.split('--')[1].split('h')[0][1:])
        run_head = int(run.name.split('--')[1].split('h')[1].split('-')[0])
        if run_component == component_name and run_layer == layer and run_head == head:
            return True
    return False


def load_head_sae(layer: int, head: int, component: str, sae_name: str = None, project_name: str = "sae-all-ioi-heads-low-l0",
                  entity: str = "mega-alignment",
                  model_project_name: str = "serimats",
                  model_entity_name: str = "georglange",
                  version: str = "v25",
                  perform_sanity_check=False) -> SparseAutoencoder:
    """
    Load an SAE from wandb that was trained on a single attention head. Runs must be named like {component}--l{layer}h{head}* or provided in sae_names.
    :param layer: The LLM layer the SAE was trained on
    :param head: The LLM head the SAE was trained on
    :param component: The LLM component the SAE was trained on, e.g. 'z', 'q', 'k', 'v'
    :param sae_name: The name or ID of the run on wandb. If None, the run will be selected based on the layer, head and component.
    :param project_name: The wandb project name
    :param entity: The wandb entity name
    :param model_project_name: The model artifact project name on wandb
    :param model_entity_name: The model artifact entity name on wandb
    :param version: The model artifact version on wandb
    :param perform_sanity_check: If True, the loaded model will be tested on a small dataset to check if it is working
    :return: A SparseAutoencoder initialized with the weights of the trained SAE
    """
    api = wandb.Api()

    if sae_name is not None:
        run_name = sae_name
    else:
        run_name = rf"{component}--l{layer}h{head}-*"

    runs = api.runs(path=f"{entity}/{project_name}")

    selected_run = next((run for run in runs if re.match(run_name, run.name) or run.id == run_name), None)

    if selected_run is None:
        error_message = f"Run {run_name} not found. Layer: {layer}, head: {head}, component: {component}"
        print(error_message)
        raise ValueError(error_message)

    artifact_name = f'model-{selected_run.id}:{version}'

    artifact = api.artifact(type="model", name=f"{model_entity_name}/{model_project_name}/{artifact_name}")
    if artifact is None:
        raise ValueError(f"Artifact not found for {model_entity_name}/{model_project_name}/{artifact_name}.")
    artifact_dir = artifact.download()

    cfg = selected_run.config

    encoder = SAETrainer.load_from_checkpoint(os.path.join(artifact_dir, 'model.ckpt'), cfg, ).sae
    if perform_sanity_check:
        sanity_check(os.path.join(artifact_dir, 'model.ckpt'), cfg, encoder)
    return encoder
