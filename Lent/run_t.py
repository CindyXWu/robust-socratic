import torch
import os
import wandb
import hydra
import argparse
import yaml
import logging
import omegaconf

from models.image_models import *
from plotting_targeted import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import * 
from train_utils import *
from configs.sweep_configs import *
from config_setup import *


teacher_dir = "teachers/"
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
parser.add_argument("--hyp_config_name", type=str, default=None)
parser.add_argument('--config_num', type=int, help='Index of experiment config in config file') # Indexes into list of dictionaries for config file
parser.add_argument("--sweep", type=bool, default=False)
args = parser.parse_args()

"""List of dictionaries - typically many of these."""
if args.config_name:
    with open(args.config_name, 'r') as f:
        configs = yaml.safe_load(f)
    config = configs[args.config_num]
# Contains: BLEGH IM DONE FOR THE DAY

"""Typically would not expect this to change between a group of runs besides via wandb sweeps, although can be made to do so in 1-1 correspondence to the experiment configs."""
if args.hyperparam_config_name:
    with open(args.hyp_config_name, 'r') as f:
        hyp_config = yaml.safe_load(f)
# Contains: lrs, epochs, batches, tau, alphas, spur
        
main_config = MainConfig
main_config.model_type, main_config.dataset_type, main_config.teach_data_num = list(ModelType)[config['teacher_num']], list(DatasetType)[config['dataset_num']], config['exp_num']
main_config.is_sweep = args.sweep


@hydra.main(config_path="configs/", config_name="defaults", version_base=None)
def main(config: MainConfig) -> None:
    """config is typed as MainConfig for duck-typing, but during runtime it's actually an OmegaConf object."""
    logging.info(f"Hydra current working directory: {os.getcwd()}")

    logger_params = {
    "name": f"{config.dataset_type}_{config.dataset.input_length}_{config.model_type}",
    "project": config.wandb_project_name,
    "settings": wandb.Settings(start_method="thread"),
    "config": omegaconf.OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
    "mode": "disabled" if not config.log_to_wandb else "online",
    }
    wandb.init(**logger_params)
    # Probably won't do sweeps over these - okay to put here relative to call to update_with_wandb_config() below
    wandb.config.dataset_type = config.dataset_type
    wandb.config.model_type = config.model_type
    
    dataset = create_or_load_dataset(config.dataset_type, config.dataset)
    train_loader, test_loader = create_dataloaders(dataset, config.dataloader)
    
    model = model_constructor(config)
    model.to(DEVICE)
    
    optimizer = optimizer_constructor(config=config, model=model)
    
    try:
        eval_frequency = config.eval_frequency if config.eval_frequency is not None else len(train_loader)
    except TypeError as e:
        msg = f"eval_frequency must be specified if using an iterable train_loader."
        raise TypeError(msg) from e
    
    epochs = math.ceil(config.num_training_iter / eval_frequency)
    train_params = {
        'model': model,
        'train_loader': train_loader,
        'test_loader': test_loader,
        'default_lr': config.optimization.default_lr,
        'epochs': epochs,
        'loss_threshold': config.loss_threshold,
        'num_eval_batches': config.num_eval_batches,
        'optimizer': optimizer,
        'project': config.wandb_project_name,
        'model_save_path': config.model_save_path,
        'device': DEVICE
    }
    train_params = update_with_wandb_config(train_params) # For wandb sweeps: update with wandb values
    train(**train_params)

    print(wandb.config)
    
    # Save teacher model and config as wandb artifacts:
    if config.save_model_as_artifact:
        model_artifact = wandb.Artifact("model", type="model", description="The trained model state_dict")
        model_artifact.add_file(".hydra/config.yaml", name="config.yaml")
        wandb.log_artifact(model_artifact)


def update_with_wandb_config(params_dict: Dict) -> Dict:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in params_dict:
        if param in wandb.config:
            print("Updating param: ", param)
            params_dict[param] = wandb.config[param]
    return params_dict


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
