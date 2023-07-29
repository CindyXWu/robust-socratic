import torch
import os
import wandb
import argparse
import yaml

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

device = "cuda" if torch.cuda.is_available() else "cpu"

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