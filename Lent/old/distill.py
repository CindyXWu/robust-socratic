import torch
import os
import wandb
import argparse
import yaml 

from datasets.image_utils import *
from info_dicts import *
from train_utils import *
from models.image_models import *
from plotting_targeted import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import *
from configs.sweep_configs import *


# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# =========================================================================
# ARGPARSE
# =========================================================================
# Add boolean flag for whether to use config file and sweep
parser = argparse.ArgumentParser()
parser.add_argument("--config_name", type=str, default=None)
# Indexes into list of dictionaries for config file
parser.add_argument('--config_num', type=int, help='Index of the configuration to use')
# This time we only do sweeps for specific losses and datasets
# (Jacobian, Contrastive and box spurious correlations)
parser.add_argument("--sweep", type=bool, default=False)
# This contains all other config things not related to distillation specific setup (things in capitals)
parser.add_argument("--main_config_name", type=str, default=None)
args = parser.parse_args()

# =========================================================================
# YAML CONFIGS
# =========================================================================
if args.config_name:
    # Load the config file - contains list of dictionaries
    with open(f'{args.config_name}', 'r') as f:
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
    wandb.config.student_mechanism = s_short_exp_name
    wandb.config.teacher_mechanism = t_short_exp_name
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]
    # Need to set the parameters you're sweeping over from wandb.configs
    tau = wandb.config.tau
    lr = wandb.config.lr
    final_lr = wandb.config.final_lr

    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, S_EXP_NUM=S_EXP_NUM, batch_size=batch_size)

    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer)


#==========================================================================
# For the following values, manual input or read from config file if one is provided
# Refer to dictionaries in info_dicts.py for what the numbers mean
#==========================================================================
is_sweep = args.sweep
T_EXP_NUM = 2
S_EXP_NUM = 4
STUDENT_NUM = 1
TEACH_NUM = 1
LOSS_NUM = 0
AUG_NUM = 0
DATASET_NUM = 2
exp_dict = exp_dict_all

if args.config_name:
    T_EXP_NUM = config['t_exp_num']
    STUDENT_NUM = config['student_num']
    TEACH_NUM = config['teacher_num']
    LOSS_NUM = config['loss_num']
    S_EXP_NUM = config['s_exp_num']
    DATASET_NUM = config['dataset_num']
base_dataset = dataset_dict[DATASET_NUM]
s_exp_name = list(exp_dict_all.keys())[S_EXP_NUM]
t_exp_name = list(exp_dict_all.keys())[T_EXP_NUM]
# "Allow only spurious mechanisms: M=100%, S1=randomized, S2=100%" ->
# M=100%, S1=randomized, S2=100%
s_short_exp_name = s_exp_name.split(":")[-1].strip()
t_short_exp_name = t_exp_name.split(":")[-1].strip()

# ======================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ======================================================================================
wandb_run = True # Set to False to check loss functions
lr = 0.3
final_lr = 0.1
temp = 30 # Fix this at about 20-30 (result of hyperparam sweeps)
epochs = 4
tau = 0.1 # Contrastive loss temperature
batch_size = 64
spurious_corr = 1
N_its = None # To standardise training length where dataset size can vary from run to run

#==============================================================================
# Stuff depending on setup params - change less often
#==============================================================================
# Necessary to make 'exp_dict' refer to correct dictionary from 'info_dicts.py'
match base_dataset:
    case 'CIFAR100':
        class_num = 100
    case 'CIFAR10':
        class_num = 10
    case 'Dominoes':
        class_num = 10
    case 'Shapes':
        class_num = 8
        N_its = 50

# Training dynamics settings depending on loss function
match LOSS_NUM:
    case 0:
        alpha = 1
    case 1:
        alpha = 0.5
    case 2:
        # Adjust for relative size of contrastive loss to distillation loss
        # E.g. 0.03 for contrastive ~60, distillation ~1
        alpha= 0.01

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
project = "Distill "+teacher_dict[TEACH_NUM]+" "+student_dict[STUDENT_NUM]+"_"+base_dataset+" gamma"
project = "test"
run_name = 'T '+teacher_dict[TEACH_NUM]+', S '+student_dict[STUDENT_NUM]+', S mech '+s_short_exp_name+', T mech '+t_short_exp_name+', Loss: '+loss_dict[LOSS_NUM]
print('project:', project)

#================================================================================
# Load saved teacher model (change only if changing file locations)
#================================================================================
# Clumsy try-except while I wrestle my codebase into sync
try:
    load_name = "teachers/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+t_short_exp_name
    checkpoint = torch.load(load_name, map_location=device)
except:
    load_name = "teachers/teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+t_short_exp_name+"_final"
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
    
    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, EXP_NUM=S_EXP_NUM, batch_size=batch_size)

    wandb.config.base_dataset = base_dataset
    wandb.config.spurious_corr = spurious_corr
    wandb.config.student_mechanism = s_short_exp_name
    wandb.config.teacher_mechanism = t_short_exp_name
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.student = student_dict[STUDENT_NUM]
    wandb.config.loss = loss_dict[LOSS_NUM]

    train_distill(teacher, student, train_loader, test_loader, base_dataset, lr, final_lr, temp, epochs, LOSS_NUM, run_name, alpha=alpha, tau=tau, s_layer=s_layer, t_layer=t_layer, N_its=N_its)
