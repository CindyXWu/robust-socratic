import torch
import os
import wandb
import argparse
import yaml
from time import gmtime, strftime

from models.resnet import wide_resnet_constructor
from models.resnet_ap import CustomResNet18, CustomResNet50
from models.lenet import LeNet5
from models.mlp import mlp_constructor
from plotting_targeted import *
from losses.jacobian import *
from losses.contrastive import *
from losses.feature_match import *
from datasets.utils_ekdeep import *
from datasets.shapes_3D import *
from info_dicts import * 
from train_utils import *
from configs.sweep_configs import *


# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

teacher_dir = "teachers/"
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(teacher_dir):
    os.makedirs(teacher_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
            "teacher": teacher_dict[TEACH_NUM],
            "dataset": "CIFAR-100",
            "batch_size": batch_size,
            "experiment": exp_name,
            },
        name=run_name
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs
    wandb.config.base_dataset = base_dataset
    wandb.config.augmentation = aug_dict[AUG_NUM]
    wandb.config.teacher = teacher_dict[TEACH_NUM]
    wandb.config.teacher_mechanism = short_exp_name

    train_loader, test_loader = create_dataloader(base_dataset=base_dataset, EXP_NUM=EXP_NUM, batch_size=batch_size, mode='train')
    base_path = teacher_dir+"teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+exp_name
    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, base_path=base_path)

#================================================================================
# SETUP PARAMS - CHANGE THESE
# Refer to dictionaries in info_dicts.py
#================================================================================
is_sweep = False
TEACH_NUM = 1
EXP_NUM = 0
AUG_NUM = 0
DATASET_NUM = 2
if args.config_name:
    EXP_NUM = config['exp_num']
    TEACH_NUM = config['teacher_num']
    DATASET_NUM = config['dataset_num']
base_dataset = dataset_dict[DATASET_NUM]
exp_name = list(exp_dict_all.keys())[EXP_NUM]
# "Allow only spurious mechanisms: M=100%, S1=randomized, S2=100%" ->
# M=100%, S1=randomized, S2=100%
short_exp_name = exp_name.split(":")[-1].strip()

#==================================================================================
# SETUP PARAMS REQUIRING MANUAL INPUT
# ===================================================================================
lr = 0.2
final_lr = 0.05
epochs = 10
batch_size = 64

# Names for wandb logging
project = "Teacher "+base_dataset
run_name = "teacher:"+teacher_dict[TEACH_NUM]+", teacher mechanism: "+short_exp_name+", aug: "+aug_dict[AUG_NUM]+" "+base_dataset
print('project:', project)

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
        epochs = 2

match TEACH_NUM:
    case 0:
        teacher = LeNet5(class_num).to(device)
    case 1:
        teacher = CustomResNet18(class_num).to(device)
    case 2:
        teacher = CustomResNet50(class_num).to(device)
    case 3:
        teacher = wide_resnet_constructor(3, class_num).to(device)

if __name__ == "__main__":
    if is_sweep:
        # Set configuration and project for sweep and initialise agent
        sweep_id = wandb.sweep(sweep=t_sweep_configuration, project=project) 
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
                "dataset": base_dataset,
                "epochs": epochs,
                "batch_size": batch_size,
                "spurious type": short_exp_name,
                "Augmentation": aug_dict[AUG_NUM]
            },
            name = run_name
        )
        wandb.config.base_dataset = base_dataset
        wandb.config.augmentation = aug_dict[AUG_NUM]
        wandb.config.teacher = teacher_dict[TEACH_NUM]
        wandb.config.teacher_mechanism = short_exp_name
        
        train_loader, test_loader = create_dataloaders(base_dataset=base_dataset, EXP_NUM=EXP_NUM, batch_size=batch_size)

        ## Plot images
        # for i, (x, y) in enumerate(train_loader):
        #     x = einops.rearrange(x, 'b c h w -> b h w c')
        #     show_images_grid(x, y, num_images=64)
        #     break

        base_path = teacher_dir+"teacher_"+teacher_dict[TEACH_NUM]+"_"+base_dataset+"_"+short_exp_name
        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, run_name, base_path=base_path)