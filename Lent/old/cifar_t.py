import torch
import os
import wandb
import math
import argparse
import yaml
from time import gmtime, strftime

from image_models import *
from plotting import *
from jacobian import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *
from info_dicts import * 
from train_utils import *

# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

output_dir = "Image_Experiments/"
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

# ======================================================================================
# ARGPARSE
# ======================================================================================
# Add boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
# Indexes into list of dictionaries for config file
parser.add_argument('--config_num', type=int, help='Index of the configuration to use')
parser.add_argument("--sweep", type=bool, default=False)
args = parser.parse_args()

# ======================================================================================
# YAML CONFIGS
# ======================================================================================
if args.config_name:
    # Load the config file - contains list of dictionaries
    with open(args.config_name, 'r') as f:
        configs = yaml.safe_load(f)
    config = configs[args.config_num]

def sweep_teacher():
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # track hyperparameters and run metadata
        config={
            "name": sweep_name,
            "teacher": teacher_dict[TEACH_NUM],
            "dataset": "CIFAR-100",
            "batch_size": batch_size,
            "experiment": cifar_exp_dict[EXP_NUM],
            },
        name=run_name
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs
    wandb.config.base_dataset = base_dataset
    wandb.config.augmentation = aug_dict[AUG_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.teacher_mechanism = cifar_exp_dict[EXP_NUM]

    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, EXP_NUM=EXP_NUM, batch_size=batch_size, mode='train')
    base_path = output_dir+"teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+cifar_exp_dict[EXP_NUM]
    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, base_path=base_path)

#================================================================================
# SETUP PARAMS - CHANGE THESE
# Refer to dictionaries in info_dicts.py
#================================================================================
is_sweep = False
TEACH_NUM = 3
EXP_NUM = 0
AUG_NUM = 0 # Define augmentation of distillation dataset
if args.config_name:
    EXP_NUM = config['experiment_num']
    TEACH_NUM = config['teacher_num']

#==============================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
#==============================================================================
lr = 0.15
final_lr = 0.01
epochs = 20
batch_size = 64
spurious_corr = 1

sweep_name = strftime("%m-%d %H:%M:%S", gmtime())
sweep_configuration = {
    'method': 'bayes',
    'name': sweep_name,
    'metric': {'goal': 'maximize', 'name': 'teacher test acc'},
    # CHANGE THESE 
    'parameters': {
        'epochs': {'values': [1]},
        'lr': {'distribution': 'log_uniform', 'min': math.log(0.1), 'max': math.log(1)},
        'final_lr': {'distribution': 'log_uniform', 'min': math.log(0.05), 'max': math.log(0.1)}
    },
    # Iter refers to number of times in code the metric is logged
    'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}

#==============================================================================
# Stuff depending on setup params - change less often
#==============================================================================
# Teacher model setup (change only if adding to dicts above)
project = "Teacher"
match TEACH_NUM:
    case 0:
        teacher = LeNet5(10).to(device)
        base_dataset = 'CIFAR10'
    case 1:
        teacher = CustomResNet18(100).to(device)
        base_dataset = 'CIFAR100'
    case 2:
        teacher = CustomResNet50(100).to(device)
        base_dataset = 'CIFAR100'
    case 3:
        teacher = wide_resnet_constructor(3, 100).to(device)
        base_dataset = 'CIFAR100'
run_name = "teacher:"+teacher_dict[TEACH_NUM]+", teacher mechanism: "+shapes_exp_dict[EXP_NUM]+", aug: "+aug_dict[AUG_NUM]+" "+base_dataset


if __name__ == "__main__":
    if is_sweep:
        # Set configuration and project for sweep and initialise agent
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project) 
        wandb.agent(sweep_id, function=sweep_teacher, count=20)
    # Should be used for retraining once best model indentified from sweep
    else:
    # Save teacher model after run
        wandb.init(
            # Set wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "dataset": base_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "spurious type": cifar_exp_dict[EXP_NUM]
            },
            name=run_name
        )

        wandb.config.base_dataset = base_dataset
        wandb.config.augmentation = aug_dict[AUG_NUM]
        wandb.config.teacher = teacher_dict[TEACH_NUM]
        wandb.config.teacher_mechanism = cifar_exp_dict[EXP_NUM]

        train_loader, test_loader = create_dataloader(base_dataset=base_dataset, EXP_NUM=EXP_NUM, batch_size=batch_size, mode='train')
        base_path = output_dir+"teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+cifar_exp_dict[EXP_NUM]
        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, base_path=base_path)