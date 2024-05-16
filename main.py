import os
import sys

import fire


def main(**kwargs):
    gpu_num = kwargs.pop("gpu_num", None)
    if gpu_num is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    # only import torch after setting the visible devices
    from training.config import SAEConfig
    from training.train import train

    config = SAEConfig(**kwargs)
    config.wandb_name = f"{config.actv_name.split('_')[-1]}-{config.wandb_name}-l{config.layer}h{config.head}-l1_{config.l1_coefficient}-{(config.train_steps * config.batch_size / 1e6):.0f}M"
    config.ckpt_name = f"{config.actv_name.split('_')[-1]}-l{config.layer}h{config.head}-l1_{config.l1_coefficient}-{(config.train_steps * config.batch_size / 1e6):.0f}M.ckpt"
    train(config)


if __name__ == "__main__":
    sys.path.append('/tmp/pycharm_project_451')
    fire.Fire(main)
