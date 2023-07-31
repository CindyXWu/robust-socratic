import torch
import os
import wandb
import hydra
from hydra.core.config_store import ConfigStore
import logging
from omegaconf import OmegaConf

from create_sweep import construct_sweep_config, load_config
from train_utils import train_distill, print_saved_model
from config_setup import DistillConfig
from constructors import model_constructor, optimizer_constructor, create_dataloaders, get_dataset_output_size


# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=DistillConfig)


@hydra.main(config_path="configs/", config_name="distill_config", version_base=None)
def main(config: DistillConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    The MainConfig class (or any dataclass you use as a type hint for the config parameter) doesn't restrict what keys can be in the configuration; it only provides additional information to your editor and to Hydra's instantiate function. The actual contents of the configuration are determined entirely by the configuration files and command line arguments.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """Command line:
    python run.py experiment@t_exp=exhaustive_0 experiment@s_exp=exhaustive_1
    """
    [t_exp_prefix, t_exp_idx] = config.t_exp.config_filename.split("_")
    [s_exp_prefix, s_exp_idx] = config.s_exp.config_filename.split("_")
    # From name field of config file
    t_exp_name, s_exp_name = config.t_exp.name.split(":")[-1].strip(), config.s_exp.name.split(":")[-1].strip()
    config.teacher_save_path = f"trained_teachers/{config.model_type}_{config.dataset_type}_{t_exp_prefix}_{t_exp_name}"
    config.student_save_path = f"trained_students/{config.model_type}_{config.dataset_type}_{s_exp_prefix}_{s_exp_name}"
    
    ## Datasets
    config.dataset.output_size = get_dataset_output_size(config)
    train_loader, test_loader = create_dataloaders(config=config)
    
    ## Models
    student = model_constructor(config).to(DEVICE)
    teacher = model_constructor(config).to(DEVICE)
    checkpoint = torch.load(config.teacher_save_path, map_location=DEVICE)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    print_saved_model(checkpoint)
    
    ## Optimizer
    epochs = config.num_iters//(len(train_loader))
    optimizer, scheduler = optimizer_constructor(config=config, model=student, train_loader=train_loader)
    
    ## wandb
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

    ## Train
    # For wandb sweeps: update with wandb values
    config = update_with_wandb_config(config)
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
    # Save student model and config as wandb artifacts:
    if config.save_model_as_artifact:
        model_artifact = wandb.Artifact("student", type="model", description="Trained student model state_dict")
        model_artifact.add_file(".hydra/distill_config.yaml", name="distill_config.yaml")
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
    config: dict = load_config('configs/distill_config.yaml')
    if config.get("sweep"):
        sweep_config = construct_sweep_config('distill_config', 'sweep_configs')
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=config.get("wandb_project_name"),
        )
        wandb.agent(sweep_id, function=main, count=10)
    else:
        main()
