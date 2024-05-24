from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class SAEConfig:
    actv_size: int = 64
    buffer_size: float = 1e7
    extraction_batch_size: int = 100
    actv_name: str = "blocks.9.attn.hook_z"
    layer: int = 9
    seq_len: int = 512
    dataset_name: str = "Skylion007/openwebtext"
    language_model: str = "gpt2-small"
    use_disc_dataset: bool = False
    store_path: str = "/var/local/glang/activations"
    store_accessor: str = "tensor"
    store_size: int = int(1e9)
    train_steps: int = 122071
    batch_size: int = 2048
    d_hidden: int = 64 * 128
    l1_coefficient: float = 0.006
    standardize_activations: bool = True
    adjust_for_dict_size: bool = False
    init_geometric_median: bool = False
    allow_lower_decoder_norm: bool = False
    disable_decoder_bias: bool = False
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    start_lr_decay: int = 0
    end_lr_decay: int = 0
    resampling_steps: list = field(default_factory=list)
    n_resampling_watch_steps: int = 5000
    lr_warmup_steps: int = 6000
    head: int = 9
    l1_exponent: int = 1
    reconstruction_loss_batch_size: int = 16
    n_resampler_samples: int = 819200
    wandb_name: str = ""
    ckpt_name: str = ""
    sae_type: Literal["vanilla", "gated"] = "vanilla"
    target_metric: str = None

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg_dict):
        return cls(**cfg_dict)

    @property
    def trainer_class(self):
        from SAETrainer import SAETrainer, GatedSAETrainer
        if self.sae_type == "vanilla":
            return SAETrainer
        if self.sae_type == "gated":
            return GatedSAETrainer
        else:
            raise ValueError(f"Unknown trainer type: {self.sae_type}")