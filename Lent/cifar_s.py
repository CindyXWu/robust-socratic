import torch
import os
import wandb
import argparse
from time import gmtime, strftime
import yaml 

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *
from image_utils import *
from info_dicts import *
from train_utils import *

# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

output_dir = "Image_Experiments/"   # Directory to store and load models from
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

## ARGPARSE ## ============================================================
# Add boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
# Indexes into list of dictionaries for config file
parser.add_argument('--config_num', type=int, help='Index of the configuration to use')
parser.add_argument("--sweep", type=bool, default=True)
args = parser.parse_args()

## OPEN YAML CONFIGS ## ===================================================
if args.config_name:
    # Load the config file - contains list of dictionaries
    with open(args.config_name, 'r') as f:
        configs = yaml.safe_load(f)
    config = configs[args.config_num]

def sweep():
    """Main function for sweep."""
    wandb.init(
        # Set wandb project where this run will be logged
        project=project,
        config={
            "dataset": "CIFAR-10",
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "final_lr": final_lr,
            "temp": temp,
            },
        name = run_name,
    )
    # alpha = wandb.config.alpha
    # wandb.config.tags = 'alpha='+str(alpha)
    spurious_corr = wandb.config.spurious_corr
    wandb.config.base_dataset = base_dataset
    wandb.config.student_mechanism = exp_dict[S_EXP_NUM]
    wandb.config.teacher_mechanism = exp_dict[T_EXP_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]

    randomize_loc = False
    match S_EXP_NUM:
        case 0:
            spurious_type = 'plain'
        case 1:
            spurious_type = 'box'
        case 2: 
            spurious_type = 'box'
            randomize_loc = True

    # Dataloaders
    train_loader = get_dataloader(load_type='train', base_dataset=base_dataset, spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)
    test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)

    # Train
    train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, randbox_test_loader, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha)

#================================================================================================
# Refer to dictionaries student_dict, exp_num, aug_dict, loss_dict, s_teach_dict in info_dicts.py
#================================================================================================
is_sweep = args.sweep
T_EXP_NUM = 2
S_EXP_NUM = 2
STUDENT_NUM = 2
TEACH_NUM = 3
LOSS_NUM = 0
AUG_NUM = 0
if args.config_name:
    T_EXP_NUM = config['t_exp_num']
    STUDENT_NUM = config['student_num']
    TEACH_NUM = config['teacher_num']
    LOSS_NUM = config['loss_num']
    S_EXP_NUM = config['s_exp_num']
## WANDB PROJECT NAME
project = "Student (debug)"
run_name = 'teacher: '+teacher_dict[TEACH_NUM]+', student: '+student_dict[STUDENT_NUM]+', student mechanism: '+exp_dict[S_EXP_NUM]+', teacher mechanism: '+exp_dict[T_EXP_NUM]+', loss: '+loss_dict[LOSS_NUM]+', aug: '+aug_dict[AUG_NUM]
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
# ======================================================================================
lr = 0.5
final_lr = 0.1
temp = 30
epochs = 15
alpha = 1 # Fraction of other distillation losses (1-alpha for distillation loss)
batch_size = 64
e_dim = 50 # embedding size for contrastive loss
spurious_corr = 1.0

sweep_method = 'grid'
sweep_count = 7
sweep_name = strftime("%m-%d %H:%M:%S", gmtime())

sweep_configuration = {
    'method': sweep_method,
    'name': sweep_name,
    'metric': {'goal': 'maximize', 'name': 'student test acc'},
    # CHANGE THESE
    'parameters': {
        'spurious_corr': {'values': [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}, # For grid search
        # 'alpha': {'distribution': 'uniform', 'min': 0, 'max': 1}, # For bayes search
    },
    # 'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}
#================================================================================

# Student model setup (change only if adding to dicts above)
match STUDENT_NUM:
    case 0:
        student = LeNet5(10).to(device)
    case 1:
        student = ResNet18_CIFAR(100).to(device)
    case 2:
        student = CustomResNet18(100).to(device)
# Teacher model setup (change only if adding to dicts above)
teacher_name = teacher_dict[TEACH_NUM]
load_path = "Image_Experiments/teacher_"+teacher_name
match TEACH_NUM:
    case 0:
        teacher = LeNet5(10).to(device)
        base_dataset = 'CIFAR10'
    case 1:
        teacher = ResNet50_CIFAR(10).to(device)
        base_dataset = 'CIFAR10'
    case 2:
        teacher = ResNet18_CIFAR(10).to(device)
        base_dataset = 'CIFAR10'
    case 3:
        teacher = ResNet18_CIFAR(100).to(device)
        base_dataset = 'CIFAR100'
    case 4:
        teacher = ResNet50_CIFAR(100).to(device)
        base_dataset = 'CIFAR100'
        
# Load saved teacher model (change only if changing file locations)
load_name = "Image_Experiments/teacher_"+teacher_name+"_"+exp_dict[T_EXP_NUM]
checkpoint = torch.load(load_name, map_location=device)
teacher.load_state_dict(checkpoint['model_state_dict'])

plain_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='plain', spurious_corr=1, randomize_loc=False)
box_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='box', spurious_corr=1, randomize_loc=False)
randbox_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='box', spurious_corr=1, randomize_loc=True)

if __name__ == "__main__":
    if is_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
        wandb.agent(sweep_id, function=sweep, count=sweep_count)
    else:
        wandb.init(
            # Set the wandb project where this run will be logged
            project=project,
            config={
                "learning_rate": lr,
                "architecture": "CNN",
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "temp": temp,
                "batch_size": batch_size,
            },
            name = run_name
        )

    randomize_loc = False
    match S_EXP_NUM:
        case 0:
            spurious_type = 'plain'
        case 1:
            spurious_type = 'box'
        case 2: 
            spurious_type = 'box'
            randomize_loc = True

    wandb.config.base_dataset = base_dataset
    wandb.config.spurious_corr = 'spurious_corr='+str(spurious_corr)
    wandb.config.student_mechanism = exp_dict[S_EXP_NUM]
    wandb.config.teacher_mechanism = exp_dict[T_EXP_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]
    
    train_loader = get_dataloader(load_type='train', base_dataset=base_dataset, spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)
    test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)

    # Train
    train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, randbox_test_loader, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha)
