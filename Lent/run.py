import torch
import os
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
import logging
from omegaconf import OmegaConf
from functools import partial

from create_sweep import construct_sweep_config, load_config
from train_utils import train_teacher
from config_setup import MainConfig
from constructors import model_constructor, optimizer_constructor, create_dataloaders, get_dataset_output_size


# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=MainConfig)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
@hydra.main(config_path="configs/", config_name="main_config", version_base=None)
def main(config: MainConfig, sweep_params: list[str] = None) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    The MainConfig class (or any dataclass you use as a type hint for the config parameter) doesn't restrict what keys can be in the configuration; it only provides additional information to your editor and to Hydra's instantiate function. The actual contents of the configuration are determined entirely by the configuration files and command line arguments.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")
    
    """Command line:
    python run.py experiment@t_exp=exhaustive_0 experiment@s_exp=exhaustive_1
    """
    # logging.info(OmegaConf.to_container(config.experiment))
    [t_exp_prefix, t_exp_idx] = config.experiment.config_filename.split("_")
    t_exp_name = config.experiment.name.split(":")[-1].strip()
    config.teacher_save_path = f"trained_teachers/{config.model_type}_{config.dataset_type}_{t_exp_prefix}_{t_exp_name.replace(' ', '_')}"
    
    ## wandb
    config.wandb_project_name = f"{config.model_type} {config.dataset_type}"
    config.wandb_run_name = f"T Mech: {t_exp_idx} {t_exp_name}"
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
    
    # Datasets
    config.dataset.output_size = get_dataset_output_size(config)
    train_loader, test_loader = create_dataloaders(config=config)

    ## Model
    teacher = model_constructor(config).to(DEVICE)
    
    ## Optimizer
    config.epochs = config.num_iters//(len(train_loader))  
    optimizer, scheduler = optimizer_constructor(config=config, model=teacher, train_loader=train_loader)
    
    ## Train
    if config.is_sweep:
        config = update_with_wandb_config(config, sweep_params) # For wandb sweeps: update with wandb values
    train_teacher(
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=DEVICE
    )
    try:
        # Save teacher model and config as wandb artifacts:
        if config.save_model_as_artifact:
            cwd = get_original_cwd()
            output_dir = HydraConfig.get().run.dir
            if HydraConfig.get().job.num is not None:
                output_dir.replace("outputs", "multirun")
            model_artifact = wandb.Artifact("teacher", type="model", description="Trained teacher model state_dict")
            model_artifact.add_file(f"{cwd}/{output_dir}/.hydra/config.yaml", name="main_config.yaml")
            wandb.log_artifact(model_artifact)
    except: # The most lazy way of saying 'this block of code is generating errors sometimes but it doesn't really matter so I'm not going to fix it'
        pass # I'm not sorry
        
    # Needed to make sure each config file in multirun initialises separately
    wandb.finish()

def update_with_wandb_config(config: OmegaConf, sweep_params: list[str]) -> OmegaConf:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in sweep_params:
        if param in wandb.config:
            print("Updating param with value from wandb config: ", param)
            OmegaConf.update(config, param, wandb.config[param], merge=True)
    return config


if __name__ == "__main__":
    """May have to edit this hard coding opening one single config file in the future."""
    config: dict = load_config('configs/main_config.yaml')
    if config.get("sweep"):
        sweep_config = construct_sweep_config('main_config', 'sweep_configs')
        sweep_params = list(sweep_config.parameters.keys())
        main_with_sweep = partial(main, config=config, sweep_params=sweep_params)
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=config.get("wandb_project_name"),
        )
        wandb.agent(sweep_id, function=main_with_sweep, count=10)
    else:
        main()
