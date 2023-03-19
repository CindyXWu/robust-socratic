import torch
import os
import wandb
import math
from time import gmtime, strftime

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *
from info_dictionaries import * 
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

shapes_dataloader = dataloader_3D_shapes('train', 1)

def sweep_teacher():
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # track hyperparameters and run metadata
        config={
            "name": sweep_name,
            "teacher": s_teacher_dict[TEACH_NUM],
            "dataset": "CIFAR-100",
            "batch_size": batch_size,
            "experiment": s_exp_dict[EXP_NUM],
            }
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs

    # match EXP_NUM:

    # Dataloaders
    train_loader = dataloader_3D_shapes('train', batch_size)
    test_loader = dataloader_3D_shapes('test', batch_size)

    # Fine-tune or train teacher from scratch
    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, project, TEACH_NUM, EXP_NUM)

# SETUP PARAMS - CHANGE THESE
#================================================================================
# Refer to dictionaries s_exp_num, aug_dict, s_teach_num in info_dictionaries.py
#================================================================================
is_sweep = False
TEACH_NUM = 2
EXP_NUM = 0
AUG_NUM = 0

# Hyperparams
lr = 0.1
final_lr = 0.01
epochs = 100
batch_size = 64
dims = [32, 32]

sweep_count = 10
sweep_method = 'bayes'
sweep_name = strftime("%m-%d %H:%M:%S", gmtime())
sweep_configuration = {
    'method': sweep_method,
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

# Teacher model setup (change only if adding to dicts above)
match TEACH_NUM:
    case 0:
        teacher = ResNet18_3Dshapes(12).to(device)
    case 1:
        teacher = ResNet18_3Dshapes(12).to(device)
    case 2:
        teacher = CustomResNet18(12).to(device)

project = s_teacher_dict[TEACH_NUM]+"_"+s_exp_dict[EXP_NUM]

if __name__ == "__main__":
    if is_sweep:
        # Set configuration and project for sweep and initialise agent
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project) 
        wandb.agent(sweep_id, function=sweep_teacher, count=sweep_count)
    # Should be used for retraining once best model indentified from sweep
    else:
    # Save teacher model after run
        wandb.init(
            # Set wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "dataset": '3D shapes',
                "epochs": epochs,
                "batch_size": batch_size,
                "spurious type": exp_dict[EXP_NUM],
                "Augmentation": aug_dict[AUG_NUM]
            }   
        )

        # match EXP_NUM:
        # Todo: add different spurious correlation experiments

        # Dataloaders
        train_loader = dataloader_3D_shapes('train', batch_size)
        test_loader = dataloader_3D_shapes('test', batch_size)

        # Fine-tune or train teacher from scratch
        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, project, TEACH_NUM, EXP_NUM, save=True)
