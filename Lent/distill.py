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
            "dataset": base_dataset,
            "teacher": teacher_dict[TEACH_NUM],
            "student": student_dict[STUDENT_NUM],
            "epochs": epochs,
            "temp": temp,
            },
        name = run_name,
    )
    # spurious_corr = wandb.config.spurious_corr
    wandb.config.spurious_corr = spurious_corr
    wandb.config.base_dataset = base_dataset
    wandb.config.student_mechanism = exp_dict[S_EXP_NUM]
    wandb.config.teacher_mechanism = exp_dict[T_EXP_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]
    tau = wandb.config.tau
    lr = wandb.config.lr
    final_lr = wandb.config.final_lr

    randomize_cue = False
    match S_EXP_NUM:
        case 0:
            cue_type = 'nocue'
        case 1:
            cue_type = 'box'
        case 2: 
            cue_type = 'box'
            randomize_cue = True
    
    train_loader = get_dataloader(load_type='train', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue)
    test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue)

    # Train
    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer)


#================================================================================================
# Refer to dictionaries student_dict, exp_num, aug_dict, loss_dict, s_teach_dict in info_dicts.py
#================================================================================================
is_sweep = args.sweep
T_EXP_NUM = 1
S_EXP_NUM = 1
STUDENT_NUM = 1
TEACH_NUM = 1
LOSS_NUM = 2
AUG_NUM = 0
base_dataset = 'Dominoes'
if args.config_name:
    T_EXP_NUM = config['t_exp_num']
    STUDENT_NUM = config['student_num']
    TEACH_NUM = config['teacher_num']
    LOSS_NUM = config['loss_num']
    S_EXP_NUM = config['s_exp_num']
    # Needs to be one of: 'CIFAR100', 'Dominoes', 'CIFAR10', 'Shapes'
    base_dataset = config['dataset']
# Necessary to make 'exp_dict' refer to correct dictionary from 'info_dicts.py'
match base_dataset:
    case 'CIFAR100':
        class_num = 100
    case 'CIFAR10':
        class_num = 10
    case 'Dominoes':
        exp_dict = dominoes_exp_dict
        class_num = 10
    case 'Shapes':
        exp_dict = shapes_exp_dict
        class_num = 8
## WANDB PROJECT NAME
project = "Distill "+teacher_dict[TEACH_NUM]+" "+student_dict[STUDENT_NUM]+"_"+base_dataset
run_name = 'T '+teacher_dict[TEACH_NUM]+', S '+student_dict[STUDENT_NUM]+', S mech '+exp_dict[S_EXP_NUM]+', T mech '+exp_dict[T_EXP_NUM]+', Loss: '+loss_dict[LOSS_NUM]
print('project:', project)

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
wandb_run = True # Set to False to check loss functions
lr = 1
final_lr = 0.3
temp = 30
epochs = 30
match LOSS_NUM:
    case 1:
        alpha = 1
    case 2:
        alpha = 0.3
tau = 0.1 # Contrastive loss temperature
batch_size = 64
spurious_corr = 1

# sweep_configuration = {
#     'method': 'bayes',
#     'name': strftime("%m-%d %H:%M:%S", gmtime()),
#     'metric': {'goal': 'maximize', 'name': 'student test acc'},
#     # CHANGE THESE
#     'parameters': {
#         #'spurious_corr': {'values': [0.5, 0.6, 0.7, 0.8, 0.9, 1]}, # For grid search
#         # 'alpha': {'distribution': 'uniform', 'min': 0, 'max': 1}, # For bayes search
#         'lr': {'distribution': 'uniform', 'min': 0.01, 'max': 0.5},
#         'final_lr': {'distribution': 'uniform', 'min': 0.001, 'max': 0.1},
#         'tau': {'distribution': 'log_uniform', 'min': -5, 'max': 2.3},
#     },
#     'early_terminate': {'type': 'hyperband', 'min_iter': 5}
# }
#================================================================================

# Student model setup (change only if adding to dicts above)
match STUDENT_NUM:
    case 0:
        student = LeNet5(class_num).to(device)
        s_layer = {'feature_extractor.10': 'feature_extractor.10'}
    case 1:
        student = CustomResNet18(class_num).to(device)
        s_layer = {'layer4.1.bn2': 'bn_bn2'}
    case 2:
        student = wide_resnet_constructor(3, class_num).to(device)
        s_layer = {"11.path2.5": "final_features"} # Contrastive feature layer

# Teacher model setup (change only if adding to dicts above)
match TEACH_NUM:
    case 1:
        teacher = CustomResNet18(class_num).to(device)
        t_layer = {'layer4.1.bn2': 'bn_bn2'}
    case 2:
        teacher = CustomResNet50(class_num).to(device)
        t_layer = {'layer4.2.bn3': 'bn_bn3'}
    case 3:
        teacher = wide_resnet_constructor(3, class_num).to(device)
        t_layer = {"11.path2.5": "final_features"} # Contrastive feature layer

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

if __name__ == "__main__":
    if is_sweep:
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
        wandb.agent(sweep_id, function=sweep, count=10)
    else:
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

    randomize_cue = False
    match S_EXP_NUM:
        case 0:
            cue_type = 'nocue'
        case 1:
            cue_type = 'box'
        case 2: 
            cue_type = 'box'
            randomize_cue = True

    wandb.config.base_dataset = base_dataset
    wandb.config.spurious_corr = spurious_corr
    wandb.config.student_mechanism = exp_dict[S_EXP_NUM]
    wandb.config.teacher_mechanism = exp_dict[T_EXP_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]
    
    train_loader = get_dataloader(load_type='train', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue)
    test_loader = get_dataloader(load_type ='test', base_dataset=base_dataset, cue_type=cue_type, cue_proportion=spurious_corr, randomize_cue=randomize_cue)

    # Train
    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer)
