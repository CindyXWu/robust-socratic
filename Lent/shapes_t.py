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
from shapes_3D import *

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

## ARGPARSE ## ============================================================
# Add boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
# Indexes into list of dictionaries for config file
parser.add_argument('--config_num', type=int, help='Index of the configuration to use')
parser.add_argument("--sweep", type=bool, default=False)
args = parser.parse_args()

## OPEN YAML CONFIGS ## ===================================================
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
            "teacher": teacher_dict[TEACH_NUM],
            "dataset": "CIFAR-100",
            "batch_size": batch_size,
            "experiment": shapes_exp_dict[EXP_NUM],
            },
        name=run_name
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs
    wandb.config.base_dataset = "3D Shapes"
    wandb.config.augmentation = aug_dict[AUG_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.teacher_mechanism = shapes_exp_dict[EXP_NUM]

    # match EXP_NUM:

    # Dataloaders
    train_loader = dataloader_3D_shapes('train', batch_size)
    test_loader = dataloader_3D_shapes('test', batch_size)

    # Fine-tune or train teacher from scratch
    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, TEACH_NUM, EXP_NUM)

# SETUP PARAMS - CHANGE THESE
#================================================================================
# Refer to dictionaries s_exp_num, aug_dict, s_teach_num in info_dictionaries.py
#================================================================================
is_sweep = False
TEACH_NUM = 3
EXP_NUM = 0
AUG_NUM = 0
if args.config_name:
    EXP_NUM = config['exp_num']
    TEACH_NUM = config['teacher_num']
run_name = "teacher:"+teacher_dict[TEACH_NUM]+", teacher mechanism: "+shapes_exp_dict[EXP_NUM]+", aug: "+aug_dict[AUG_NUM]+" shapes"

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
lr = 0.5
final_lr = 0.08
epochs = 8
batch_size = 64

sweep_configuration = {
    'method': 'bayes',
    'name': strftime("%m-%d %H:%M:%S", gmtime()),
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
shapes_exp_dict = {0: "Shape_Color", 1: "Shape_Color_Floor", 2: "Shape_Color_Scale", 3: "Floor", 4: "Color", 5: "Floor_Color"}
#==============================================================================
# Teacher model setup (change only if adding to dicts above)
project = "Teacher Shapes"
match TEACH_NUM:
    case 1:
        teacher = CustomResNet18(12).to(device)
        dataset = "Shapes"
    case 2:
        teacher = CustomResNet50(12).to(device)
        dataset = "Shapes"
    case 3:
        teacher = wide_resnet_constructor(3, 12).to(device)
        dataset = "Shapes"

if __name__ == "__main__":
    if is_sweep:
        # Set configuration and project for sweep and initialise agent
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project) 
        wandb.agent(sweep_id, function=sweep_teacher, count=10)
    # Should be used for retraining once best model indentified from sweep
    else:
    # Save teacher model after run
        wandb.init(
            # Set wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config={
                "LR": lr,
                "final LR": final_lr,
                "dataset": '3D shapes',
                "epochs": epochs,
                "batch_size": batch_size,
                "spurious type": shapes_exp_dict[EXP_NUM],
                "Augmentation": aug_dict[AUG_NUM]
            },
            name = run_name
        )
        wandb.config.base_dataset = "3D Shapes"
        wandb.config.augmentation = aug_dict[AUG_NUM]
        wandb.config.teacher = teacher_dict[TEACH_NUM]
        wandb.config.teacher_mechanism = shapes_exp_dict[EXP_NUM]
        
        randomise = False
        match EXP_NUM:
            case 1:
                mechanisms = [0]
            case 2:
                mechanisms = [3]
            case 3:
                randomise = True
                mechanisms = [0]
            case 4:
                randomise = True
                mechanisms = [3]
            case 5:
                randomise = True
                mechanisms = [0, 3]

        train_loader = dataloader_3D_shapes('train', batch_size, randomise=randomise, mechanisms=mechanisms)
        test_loader = dataloader_3D_shapes('test', batch_size, randomise=randomise, mechanisms=mechanisms)

        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, TEACH_NUM, EXP_NUM, dataset=dataset)
