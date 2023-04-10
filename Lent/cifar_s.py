import torch
import os
import wandb
import argparse
from time import gmtime, strftime
import yaml 

from image_models import *
from plotting import *
from jacobian import *
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
# This time we only do sweeps for specific losses and datasets
# (Jacobian, Contrastive and box spurious correlations)
parser.add_argument("--sweep", type=bool, default=False)
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
T_EXP_NUM = 1
S_EXP_NUM = 1
STUDENT_NUM = 2
TEACH_NUM = 3
LOSS_NUM = 2
AUG_NUM = 0
if args.config_name:
    T_EXP_NUM = config['t_exp_num']
    STUDENT_NUM = config['student_num']
    TEACH_NUM = config['teacher_num']
    LOSS_NUM = config['loss_num']
    S_EXP_NUM = config['s_exp_num']
## WANDB PROJECT NAME
project = teacher_dict[TEACH_NUM]+" "+student_dict[STUDENT_NUM]
run_name = 'T '+teacher_dict[TEACH_NUM]+', S '+student_dict[STUDENT_NUM]+', S mech '+exp_dict[S_EXP_NUM]+', T mech '+exp_dict[T_EXP_NUM]+', Loss: '+loss_dict[LOSS_NUM]+', Aug: '+aug_dict[AUG_NUM]

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
wandb_run = True # Set to False to check loss functions
lr = 0.5
final_lr = 0.05
temp = 30
epochs = 20
alpha = 1 # Fraction of other distillation losses (1-alpha for distillation loss)
batch_size = 64
spurious_corr = 1

sweep_configuration = {
    'method': 'grid',
    'name': strftime("%m-%d %H:%M:%S", gmtime()),
    'metric': {'goal': 'maximize', 'name': 'student test acc'},
    # CHANGE THESE
    'parameters': {
        'spurious_corr': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}, # For grid search
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
        student = CustomResNet18(100).to(device)
    case 2:
        student = wide_resnet_constructor(3, 100).to(device)

# Teacher model setup (change only if adding to dicts above)
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

# Load saved teacher model (change only if changing file locations)
# Clumsy try-except while I wrestle my codebase into sync
try:
    load_name = "Image_Experiments/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+exp_dict[T_EXP_NUM]
    checkpoint = torch.load(load_name, map_location=device)
except:
    load_name = "Image_Experiments/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+exp_dict[T_EXP_NUM]+"_final"
checkpoint = torch.load(load_name, map_location=device)
teacher.load_state_dict(checkpoint['model_state_dict'])
test_acc = checkpoint['test_acc']
print("Loaded teacher model with test accuracy: ", test_acc[-1])

plain_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='plain', spurious_corr=1, randomize_loc=False)
box_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='box', spurious_corr=1, randomize_loc=False)
randbox_test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, spurious_type='box', spurious_corr=1, randomize_loc=True)

if __name__ == "__main__":
    if is_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
        wandb.agent(sweep_id, function=sweep, count=10)
    elif wandb_run:
        wandb.init(
            # Set the wandb project where this run will be logged
            project=project,
            config={
                "start LR": lr,
                "final LR": final_lr,
                "dataset": base_dataset,
                "epochs": epochs,
                "temp": temp,
                "teacher_acc": test_acc[-1],
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

    if wandb_run:
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
