import torch
import os
import wandb
import hydra
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
import argparse
import logging
from omegaconf import OmegaConf

from models.resnet import wide_resnet_constructor
from models.resnet_ap import CustomResNet18
from models.lenet import LeNet5
from models.mlp import mlp_constructor
from plotting_targeted import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import * 
from train_utils import create_dataloaders, train_teacher, train_distill
from configs.sweep_configs import *
from config_setup import *
from constructors import *


# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=MainConfig)


@hydra.main(config_path="configs/", config_name="main_config", version_base=None)
def main(config: MainConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    The MainConfig class (or any dataclass you use as a type hint for the config parameter) doesn't restrict what keys can be in the configuration; it only provides additional information to your editor and to Hydra's instantiate function. The actual contents of the configuration are determined entirely by the configuration files and command line arguments.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")

    """Command line:
    python run.py experiment@t_exp=exhaustive_0 experiment@s_exp=exhaustive_1
    """
    t_exp_idx = config.t_exp.name.split("_")[-1]
    s_exp_idx = config.s_exp.name.split("_")[-1]
    t_exp_name = config.t_exp.name.split(":")[-1].strip()
    s_exp_name = config.s_exp.name.split(":")[-1].strip()
    
    train_loader, test_loader = create_dataloaders(config=config)
    
    if config.is_distill:
        student = model_constructor(config).to(DEVICE)
        teacher = model_constructor(config).to(DEVICE)
        optimizer, scheduler = optimizer_constructor(config=config, model=student)
        config.wandb_run_name = f"T Mech: {t_exp_idx} {t_exp_name}"
    else:
        teacher = model_constructor(config).to(DEVICE)
        optimizer, scheduler = optimizer_constructor(config=config, model=teacher)
        config.wandb_run_name = f"T Mech: {t_exp_idx} {t_exp_name}, S Mech: {s_exp_idx} {s_exp_name}, Loss: {config.distill_loss_type}"
    
    logger_params = {
    "name": config.wandb_run_name,
    "project": config.wandb_project_name,
    "settings": wandb.Settings(start_method="thread"),
    "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    "mode": "disabled" if not config.log_to_wandb else "online",
    }
    wandb.init(**logger_params)
    # Probably won't do sweeps over these - okay to put here relative to call to update_with_wandb_config() below
    wandb.config.dataset_type = config.dataset_type
    wandb.config.model_type = config.model_type

    epochs = config.distill_iters//(len(train_loader))  
    config = update_with_wandb_config(config) # For wandb sweeps: update with wandb values
    if config.is_distill:
        train_distill(
            teacher=teacher,
            student=student,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epochs=epochs,
            device=DEVICE
        )
    else:
        train_teacher(
            model=teacher,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epochs=epochs,
            device=DEVICE
        )
        # Save teacher model and config as wandb artifacts:
        if config.save_model_as_artifact:
            model_artifact = wandb.Artifact("model", type="model", description="The trained model state_dict")
            model_artifact.add_file(".hydra/config.yaml", name="config.yaml")
            wandb.log_artifact(model_artifact)


def update_with_wandb_config(config: OmegaConf) -> OmegaConf:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in OmegaConf.to_container(config):
        if param in wandb.config:
            print("Updating param with value from wandb config: ", param)
            OmegaConf.update(config, param, wandb.config[param], merge=True)
    return config


if __name__ == "__main__":
    """May have to edit this hard coding opening one single config file in the future."""
    config: Dict = load_config('configs/defaults.yaml')
    if config.get("sweep"):
        sweep_config = construct_sweep_config('defaults', 'sweep_configs')
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=config.get("wandb_project_name"),
        )
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()
