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
            "experiment": dominoes_exp_dict[EXP_NUM],
            },
        name=run_name
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs
    wandb.config.base_dataset = "Dominoes"
    wandb.config.augmentation = aug_dict[AUG_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.teacher_mechanism = dominoes_exp_dict[EXP_NUM]

    randomize_cues = [False, False]
    randomize_img = False
    match EXP_NUM:
        case 0:
            cue_proportions = [0.0, 0.0]
        case 1:
            cue_proportions = [mnist_frac, 0.0]
            randomize_img = True
        case 2:
            cue_proportions = [0.0, box_frac]
            randomize_img = True
        case 3:
            randomize_img = True
            cue_proportions = [mnist_frac, box_frac]
        case 4:
            cue_proportions = [mnist_frac, 0.0]
        case 5:
            cue_proportions = [0.0, box_frac]
        case 6:
            cue_proportions = [mnist_frac, box_frac]

    train_loader = get_dataloader(load_type='train', base_dataset='Dominoes Box', batch_size=64, randomize_img = randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)
    test_loader = get_dataloader(load_type='test', base_dataset='Dominoes Box', batch_size=64, randomize_img = randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)

    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, TEACH_NUM, EXP_NUM, dataset=dataset)


# SETUP PARAMS - CHANGE THESE
#================================================================================
# Refer to dictionaries s_exp_num, aug_dict, s_teach_num in info_dictionaries.py
#================================================================================
is_sweep = False
TEACH_NUM = 3
EXP_NUM = 6
AUG_NUM = 0
if args.config_name:
    EXP_NUM = config['exp_num']
    TEACH_NUM = config['teacher_num']
run_name = "teacher:"+teacher_dict[TEACH_NUM]+", teacher mechanism: "+dominoes_exp_dict[EXP_NUM]+", aug: "+aug_dict[AUG_NUM]+" shapes"

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
lr = 0.5
final_lr = 0.08
epochs = 8
batch_size = 64
mnist_frac = 1.0
box_frac = 1.0

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
#dominoes_exp_dict = {0: "CIFAR10", 1: "MNIST", 2: "Box", 3: "MNIST_Box", 4: "CIFAR10_MNIST", 5: "CIFAR10_Box", 6: "CIFAR10_MNIST_Box"}
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
                "spurious type": dominoes_exp_dict[EXP_NUM],
                "Augmentation": aug_dict[AUG_NUM]
            },
            name = run_name
        )
        wandb.config.base_dataset = "Dominoes"
        wandb.config.augmentation = aug_dict[AUG_NUM]
        wandb.config.teacher = teacher_dict[TEACH_NUM]
        wandb.config.teacher_mechanism = dominoes_exp_dict[EXP_NUM]
        
        randomize_cues = [False, False]
        randomize_img = False
        match EXP_NUM:
            case 0:
                cue_proportions = [0.0, 0.0]
            case 1:
                cue_proportions = [mnist_frac, 0.0]
                randomize_img = True
            case 2:
                cue_proportions = [0.0, box_frac]
                randomize_img = True
            case 3:
                randomize_img = True
                cue_proportions = [mnist_frac, box_frac]
            case 4:
                cue_proportions = [mnist_frac, 0.0]
            case 5:
                cue_proportions = [0.0, box_frac]
            case 6:
                cue_proportions = [mnist_frac, box_frac]

        train_loader = get_dataloader(load_type='train', base_dataset='Dominoes Box', batch_size=64, randomize_img = randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)
        test_loader = get_dataloader(load_type='test', base_dataset='Dominoes Box', batch_size=64, randomize_img = randomize_img, cue_proportions=cue_proportions, randomize_cues=randomize_cues)

        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, TEACH_NUM, EXP_NUM, dataset=dataset)
