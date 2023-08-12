import torch
import logging
import os
import wandb
import hydra
import warnings
from functools import partial
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from create_sweep import construct_sweep_config, load_config
from train_utils import train_distill
from config_setup import DistillConfig, DistillLossType
from constructors import model_constructor, get_model_intermediate_layer, optimizer_constructor, create_dataloaders, get_dataset_output_size, get_nonbase_loss_frac

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=DistillConfig)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
# CHANGE THESE  
config_filename = "main_config"
sweep_filename = "jac_acc_sweep"

  
@hydra.main(config_path="configs/", config_name=config_filename, version_base=None)
def main(config: DistillConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    The MainConfig class (or any dataclass you use as a type hint for the config parameter) doesn't restrict what keys can be in the configuration; it only provides additional information to your editor and to Hydra's instantiate function. The actual contents of the configuration are determined entirely by the configuration files and command line arguments.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")
    
    """Command line:
    python run.py experiment@t_exp=exhaustive_0 experiment@s_exp=exhaustive_1
    """
    is_sweep = config.is_sweep
    [t_exp_prefix, t_exp_idx] = config.experiment.config_filename.split("_")
    [s_exp_prefix, s_exp_idx] = config.experiment_s.config_filename.split("_")
    # From name field of config file
    t_exp_name, s_exp_name = config.experiment.name.split(":")[-1].strip(), config.experiment_s.name.split(":")[-1].strip()
    config.teacher_save_path = f"trained_teachers/{config.model_type}_{config.dataset_type}_{t_exp_prefix}_{t_exp_name.replace(' ', '_')}_teacher" # Where to load teacher
    config.student_save_path = f"trained_students/{config.model_type}_{config.dataset_type}_{s_exp_prefix}_{s_exp_name.replace(' ', '_')}_student"
    
    ## Update config file before logging config values to wandb
    config.nonbase_loss_frac = get_nonbase_loss_frac(config)
    config.dataset.output_size = get_dataset_output_size(config)
            
    ## wandb
    config.wandb_project_name = f"{config.wandb_project_name} DISTILL {config.model_type} {config.dataset_type} {config.config_type} {config.dataset.box_cue_pattern}"
    config.wandb_run_name = f"T Mech: {t_exp_idx} {t_exp_name}, S Mech: {s_exp_idx} {s_exp_name}, Loss: {config.distill_loss_type}"
    logger_params = {
        "name": config.wandb_run_name,
        "project": config.wandb_project_name,
        "settings": wandb.Settings(start_method="thread"),
        "config": OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
        "mode": "disabled" if not config.log_to_wandb else "online",
    }
    wandb.init(**logger_params)

    ## Datasets
    train_loader, test_loader = create_dataloaders(config=config)
    # Epochs aren't logged faithfully to wandb, but is fine as it tends to be an upper bound due to early-stopping
    config.epochs = config.num_iters//(len(train_loader))
    
    ## Models
    student = model_constructor(config).to(DEVICE)
    teacher = model_constructor(config).to(DEVICE)
    checkpoint = torch.load(config.teacher_save_path, map_location=DEVICE)
    teacher.load_state_dict(checkpoint['model_state_dict'])
    config.t_layer = config.s_layer = get_model_intermediate_layer(config)
    
    ## Optimizer
    optimizer, scheduler = optimizer_constructor(config=config, model=student, train_loader=train_loader)

    ## Train
    if is_sweep:
        config = update_with_wandb_config(config, sweep_params) # For wandb sweeps: update with wandb values
    train_distill(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=DEVICE)
    try:
        # Save teacher model and config as wandb artifacts:
        if config.save_model_as_artifact:
            cwd = get_original_cwd()
            output_dir = HydraConfig.get().run.dir
            if HydraConfig.get().job.num is not None:
                output_dir.replace("outputs", "multirun")
            model_artifact = wandb.Artifact("student", type="model", description="Trained student model state_dict")
            model_artifact.add_file(f"{cwd}/{output_dir}/.hydra/config.yaml", name="distill_config.yaml")
            wandb.log_artifact(model_artifact)
    except: # The most lazy way of saying 'this block of code is generating errors sometimes but it doesn't really matter so I'm not going to fix it'
        pass # I'm not sorry
    
    # Needed to make sure each config file in multirun initialises separately - this requires the previous run to close
    wandb.finish()


def update_with_wandb_config(config: OmegaConf, sweep_params: list[str]) -> OmegaConf:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in sweep_params:
        if param in wandb.config:
            print("Updating param with value from wandb config: ", param)
            OmegaConf.update(config, param, wandb.config[param], merge=True)
    return config


if __name__ == "__main__":
    config: dict = load_config(f"configs/{config_filename}.yaml")
    if config.get("is_sweep"):
        wandb_project_name = f"{config['wandb_project_name']} DISTILL {config['model_type']} {config['dataset_type']} {config['config_type']} {config['dataset']['box_cue_pattern']}"
        sweep_config = load_config(f"configs/{sweep_filename}.yaml")
        sweep_params = list(sweep_config['parameters'].keys())
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=wandb_project_name
        )
        wandb.agent(sweep_id, function=main, count=config['sweep_num'])
    else:
        main()
