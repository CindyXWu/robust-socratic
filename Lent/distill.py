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

output_dir = "Image_Experiments/"   # Directory to store and load models from
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# ======================================================================================
# ARGPARSE
# ======================================================================================
# Add boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
# Indexes into list of dictionaries for config file
parser.add_argument('--config_num', type=int, help='Index of the configuration to use')
# This time we only do sweeps for specific losses and datasets
# (Jacobian, Contrastive and box spurious correlations)
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

def sweep():
    """Function needed for wandb sweep. Boilerplate repeated code from main()."""
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
    # Need to set the parameters you're sweeping over from wandb.configs
    tau = wandb.config.tau
    lr = wandb.config.lr
    final_lr = wandb.config.final_lr

    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, S_EXP_NUM=S_EXP_NUM, batch_size=batch_size, spurious_corr=spurious_corr, mode='train')

    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer)


#================================================================================================
# For the following values, manual input or read from config file if one is provided
# Refer to dictionaries in info_dicts.py for what the numbers mean
#================================================================================================
is_sweep = args.sweep
T_EXP_NUM = 0
S_EXP_NUM = 1
STUDENT_NUM = 1
TEACH_NUM = 1
LOSS_NUM = 1
AUG_NUM = 0
DATASET_NUM = 2

if args.config_name:
    T_EXP_NUM = config['t_exp_num']
    STUDENT_NUM = config['student_num']
    TEACH_NUM = config['teacher_num']
    LOSS_NUM = config['loss_num']
    S_EXP_NUM = config['s_exp_num']
    DATASET_NUM = config['dataset']
base_dataset = dataset_dict[DATASET_NUM]

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
wandb_run = True # Set to False to check loss functions
lr = 0.01
final_lr = 0.05
temp = 30 # Fix this at about 20-30 (result of hyperparam sweeps)
epochs = 10
tau = 0.1 # Contrastive loss temperature
batch_size = 64
spurious_corr = 1

## Uncomment and change values for sweep
# Note I have an early stop criteria but this is optional
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

#==============================================================================
# Stuff depending on setup params - change less often
#==============================================================================
# Necessary to make 'exp_dict' refer to correct dictionary from 'info_dicts.py'
match base_dataset:
    case 'CIFAR100':
        class_num = 100
        exp_dict = cifar_exp_dict
    case 'CIFAR10':
        class_num = 10
        exp_dict = cifar_exp_dict
    case 'Dominoes':
        exp_dict = dominoes_exp_dict
        class_num = 10
    case 'Shapes':
        exp_dict = shapes_exp_dict
        class_num = 8

# Training dynamics settings depending on loss function
match LOSS_NUM:
    case 0:
        alpha, lr, final_lr = 1, 0.1, 0.01
    case 1:
        alpha, lr, final_lr = 0.5, 0.5, 0.05
        epochs = 20
    case 2:
        # Adjust for relative size of contrastive loss to distillation loss
        # E.g. 0.03 for contrastive ~60, distillation ~1
        alpha, lr, final_lr = 0.01, 0.5, 0.05
        epochs = 20

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
    case 0:
        teacher = LeNet5(class_num).to(device)
        t_layer = {'feature_extractor.10': 'feature_extractor.10'}
    case 1:
        teacher = CustomResNet18(class_num).to(device)
        t_layer = {'layer4.1.bn2': 'bn_bn2'}
    case 2:
        teacher = CustomResNet50(class_num).to(device)
        t_layer = {'layer4.2.bn3': 'bn_bn3'}
    case 3:
        teacher = wide_resnet_constructor(3, class_num).to(device)
        t_layer = {"11.path2.5": "final_features"} # Contrastive feature layer

# Names for wandb logging
project = "Distill "+teacher_dict[TEACH_NUM]+" "+student_dict[STUDENT_NUM]+"_"+base_dataset
run_name = 'T '+teacher_dict[TEACH_NUM]+', S '+student_dict[STUDENT_NUM]+', S mech '+exp_dict[S_EXP_NUM]+', T mech '+exp_dict[T_EXP_NUM]+', Loss: '+loss_dict[LOSS_NUM]
print('project:', project)

#================================================================================
# Load saved teacher model (change only if changing file locations)
#================================================================================
# Clumsy try-except while I wrestle my codebase into sync
try:
    load_name = "Image_Experiments/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+exp_dict[T_EXP_NUM]
    checkpoint = torch.load(load_name, map_location=device)
except:
    load_name = "Image_Experiments/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+exp_dict[T_EXP_NUM]+"_final"
checkpoint = torch.load(load_name, map_location=device)
teacher.load_state_dict(checkpoint['model_state_dict'])
try:
    test_acc = checkpoint['test_acc']
    print("Loaded teacher model with test accuracy: ", test_acc[-1])
except:
    test_acc = [0]


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
    
    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, EXP_NUM=S_EXP_NUM, batch_size=batch_size, spurious_corr=spurious_corr, mode='train')

    wandb.config.base_dataset = base_dataset
    wandb.config.spurious_corr = spurious_corr
    wandb.config.student_mechanism = exp_dict[S_EXP_NUM]
    wandb.config.teacher_mechanism = exp_dict[T_EXP_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]

    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer)
