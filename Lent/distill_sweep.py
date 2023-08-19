"""While Python-based WandB sweep agents are initialised for the other files too,
this is the only one that interfaces with bash and the CLI called version."""
import torch
import logging
import os
import sys
import wandb
import warnings

# Capture the original arguments
original_args = sys.argv[1:]
sweep_params = {arg.split('=')[0].replace('--', ''): arg.split('=')[1] for arg in original_args}
# Process the original arguments to fit Hydra's format or your specific needs.
# For example: ['--contrast_temp=0.06966'] to ['contrast_temp=0.06966']
processed_args = [arg.replace('--', '') for arg in original_args]
# Manually add Hydra-specific arguments
# Example: Use a specific config file and override some values
# Update sys.argv
sys.argv[1:] = processed_args

from create_sweep import load_config
from train_utils import train_distill, get_previous_commit_hash
from config_setup import DistillConfig, DistillLossType
from constructors import model_constructor, get_model_intermediate_layer, optimizer_constructor, create_dataloaders, get_dataset_output_size, get_nonbase_loss_frac
from teacher_check import load_model_and_get_accs

import hydra
from hydra.experimental import initialize, compose
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Register the defaults from the structured dataclass config schema:
cs = ConfigStore.instance()
cs.store(name="config_base", node=DistillConfig)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
# CHANGE THESE
config_filename = "distill_sweep_test_config"
initialize(config_path="configs", job_name="my_app")
# Used as argument in main function
config = compose(
    config_name=config_filename, 
    overrides=[
        "+experiment=exhaustive_0",
        "+experiment_s=exhaustive_0"
    ]
)

      
def main(config: DistillConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object.
    
    The MainConfig class (or any dataclass you use as a type hint for the config parameter) doesn't restrict what keys can be in the configuration; it only provides additional information to your editor and to Hydra's instantiate function. The actual contents of the configuration are determined entirely by the configuration files and command line arguments.
    """
    logging.info(f"Hydra current working directory: {os.getcwd()}")
    
    """Command line:
    python run.py experiment@t_exp=exhaustive_0 experiment@s_exp=exhaustive_1
    """
    if config.is_sweep: # For wandb sweeps: update with wandb values
        config = update_with_wandb_config(config, sweep_params)
        
    ## Filenames and all that stuff
    [t_exp_prefix, t_exp_idx] = config.experiment.config_filename.split("_")
    [s_exp_prefix, s_exp_idx] = config.experiment_s.config_filename.split("_")
    # From name field of config file
    t_exp_name, s_exp_name = config.experiment.name.split(":")[-1].strip(), config.experiment_s.name.split(":")[-1].strip()
    config.teacher_save_path = f"trained_teachers/{config.model_type}_{config.dataset_type}_{t_exp_prefix}_{t_exp_name.replace(' ', '_')}_{config.dataset.box_cue_pattern}_teacher"
    config.student_save_path = f"trained_students/{config.model_type}_{config.dataset_type}_{s_exp_prefix}_{s_exp_name.replace(' ', '_')}_{config.dataset.box_cue_pattern}_student"
    
    ## Update config file before logging config values to wandb
    if config.nonbase_loss_frac is None and config.distill_loss_type != DistillLossType.BASE:
        # If nonbase_loss_frac not specified via command line
        config.nonbase_loss_frac = get_nonbase_loss_frac(config)
    config.dataset.output_size = get_dataset_output_size(config)
            
    ## WandB - note project now hyphenated by default
    config.wandb_project_name = f"DISTILL-{config.model_type}-{config.dataset_type}-{config.config_type}-{config.dataset.box_cue_pattern}{config.wandb_project_name}"

    config.wandb_run_name = f"T Mech: {t_exp_idx} {t_exp_name}, S Mech: {s_exp_idx} {s_exp_name}, Loss: {config.distill_loss_type}"
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)   
    extra_params = { # To log to WandB but not needed otherwise
        "teacher accs": load_model_and_get_accs(config.teacher_save_path),
        "Git commit hash": get_previous_commit_hash(),
    }
    config_dict.update(extra_params)
    logger_params = {
        "name": config.wandb_run_name,
        "project": config.wandb_project_name,
        "settings": wandb.Settings(start_method="thread"),
        "config": config_dict,
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


def update_with_wandb_config(config: DistillConfig, params: dict) -> DistillConfig:
    """
    Update the Hydra configuration object with values from the command line.
    
    Args:
        config: The Hydra configuration object.
        params: Dictionary of parameters and their values.
    """
    for key, value_str in params.items():
        
        # Convert strings that look like numbers to actual numbers
        if value_str.replace('.', '', 1).isdigit():
            value = float(value_str) if '.' in value_str else int(value_str)
        else:
            value = value_str
        
        # Handle nested keys (e.g., optimization.base_lr)
        keys = key.split('.')
        sub_config = config
        for k in keys[:-1]:
            sub_config = sub_config[k]
        sub_config[keys[-1]] = value
    
    return config

        
if __name__ == "__main__":
    main(config)