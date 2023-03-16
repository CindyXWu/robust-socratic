import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import os
import wandb
import math
from tqdm import tqdm
from time import gmtime, strftime
import yaml 

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *
from image_utils import *

# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

output_dir = "Image_Experiments/"   # Directory to store and load models from
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

with open('configs.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# for experiment in configs:
#     print(f"Running {experiment['experiment_name']} with learning rate {experiment['learning_rate']} and batch size {experiment['batch_size']}")

def load_cifar_10(dims):
    """Load CIFAR-10 dataset and return dataloaders.
    :param dims: tuple, dimensions of the images
    """
    transform = transforms.Compose([transforms.Resize((dims[0],dims[1])),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,  
                                0.406], [0.229, 0.224, 0.225])])
    trainset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    return trainset, testset, trainloader, testloader
    
@torch.no_grad()
def evaluate(model, dataset, batch_size, max_ex=0):
    """Evaluate model accuracy on dataset."""
    acc = 0
    for i, (features, labels) in enumerate(dataset):
        labels = labels.to(device)
        features = features.to(device)
        # Batch size in length, varying from 0 to 1
        scores = nn.functional.softmax(model(features.to(device)), dim=1)
        _, pred = torch.max(scores, 1)
        # Save to pred 
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # Return average accuracy as a percentage
    # Fraction of data points correctly classified
    return (acc*100 / ((i+1)*batch_size))

def weight_reset(model):
    """Reset weights of model at start of training."""
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Instantiate losses
kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='mean')

def base_distill_loss(scores, targets, temp):
    scores = scores/temp
    targets = F.softmax(targets/temp).argmax(dim=1)
    return ce_loss(scores, targets)

def train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, ranbox_test_loader, lr, final_lr, temp, epochs, repeats, loss_num, alpha=None):
    """Train student model with distillation loss.
    
    Includes LR scheduling. Change loss function as required. 
    N.B. I need to refator this at some point.
    """
    for _ in range(repeats):
        optimizer = optim.SGD(student.parameters(), lr=lr)
        scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))
        it = 0
        train_acc = []
        test_acc = []
        train_loss = []  # loss at iteration 0
        weight_reset(student)

        for epoch in range(epochs):
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)
                scores = student(inputs)
                targets = teacher(inputs)

                input_dim = 32*32*3
                output_dim = scores.shape[1]
                batch_size = inputs.shape[0]

                # for param in student.parameters():
                #     assert param.requires_grad
                match loss_num:
                    case 0: # Base distillation loss
                        loss = base_distill_loss(scores, targets, temp)
                    case 1: # Jacobian loss
                        input_dim = 32*32*3
                        output_dim = scores.shape[1]
                        batch_size = inputs.shape[0]
                        loss = jacobian_loss(scores, targets, inputs, T=1, alpha=1, batch_size=batch_size, loss_fn=mse_loss, input_dim=input_dim, output_dim=output_dim)
                    case 2: # Feature map loss - currently only for self-distillation
                        layer = 'feature_extractor.10'
                        s_map = student.attention_map(inputs, layer)
                        t_map = teacher.attention_map(inputs, layer).detach()
                        loss = feature_map_diff(scores, targets, s_map, t_map, T=1, alpha=0.2, loss_fn=mse_loss, aggregate_chan=False)
                    case 3: # Attention Jacobian loss
                        loss = jacobian_attention_loss(student, teacher, scores, targets, inputs, batch_size, T=1, alpha=0.8, loss_fn=kl_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                lr = scheduler.get_lr()
                train_loss.append(loss.detach().cpu().numpy())

                # if it == 0:
                #     # Check that model is training correctly
                #     for param in student.parameters():
                #         assert param.grad is not None
                if it % 100 == 0:
                    batch_size = inputs.shape[0]
                    train_acc.append(evaluate(student, train_loader, batch_size, max_ex=100))
                    test_acc.append(evaluate(student, test_loader, batch_size))
                    plain_acc = evaluate(student, plain_test_loader, batch_size)
                    box_acc = evaluate(student, box_test_loader, batch_size)
                    randbox_acc = evaluate(student, ranbox_test_loader, batch_size)
                    print('Iteration: %i, %.2f%%' % (it, test_acc[-1]), "Epoch: ", epoch, "Loss: ", train_loss[-1])
                    print("Project {}, LR {}, temp {}".format(project, lr, temp))
                    wandb.log({"student train acc": train_acc[-1], "student test acc": test_acc[-1], "student plain training test acc": plain_acc, "student spurious box test acc": box_acc, "student randomised spurious box test acc": randbox_acc, "student loss": train_loss[-1], 'student lr': lr})
                it += 1

def sweep():
    """Main function for sweep."""
    wandb.init(
        # Set wandb project where this run will be logged
        project=project,
        config={
            "dataset": "CIFAR-10",
            "batch_size": batch_size,
            }
    )
    alpha = wandb.config.alpha
    # spurious_corr = wandb.config.spurious_corr

    randomize_loc = False
    spurious_corr = 1.0
    name = exp_dict[EXP_NUM]    # Dataset saved name - see dict above
    match EXP_NUM:
        case 0:
            spurious_type = 'plain'
        case 1:
            spurious_type = 'box'
        case 2: 
            spurious_type = 'box'
            randomize_loc = True
        case 3:
            spurious_type = 'box'
            spurious_corr = 0.5
        case 4:
            spurious_type = 'box'
            randomize_loc = True
            spurious_corr = 0.5

    # Dataloaders
    train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)
    test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)

    # Train
    train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, randbox_test_loader, lr, final_lr, temp, epochs, 1, LOSS_NUM, alpha=alpha)

# Look at these as reference to set values for above variables
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}
student_dict = {0: "LeNet5_CIFAR10"}
teacher_dict = {0: "LeNet5_CIFAR10", 1: "ResNet50_CIFAR10", 2: "ResNet18_CIFAR10"}
loss_dict  = {0: "Base Distillation", 1: "Jacobian", 2: "Feature Map", 3: "Attention Jacobian"}

# SETUP PARAMS - CHANGE THESE
#================================================================================
#================================================================================
is_sweep = True
EXP_NUM = 1
STUDENT_NUM = 0
TEACH_NUM = 0
LOSS_NUM = 1

# Hyperparams
lr = 0.3
final_lr = 0.08
temp = 30
epochs = 1
alpha = 0.5 # Fraction of other distillation losses (1-alpha for distillation loss)
batch_size = 64
dims = [32, 32]
sweep_method = 'grid'
sweep_count = 7
sweep_name = 'Feature map match for alpha' + strftime("%m-%d %H:%M:%S", gmtime())
e_dim = 50 # embedding size for contrastive loss
repeats = 1 # I don't think I will use this - repeats will be done by calling this script multiple times

wandb.config.tags = ['no_spurious', '0,5', '0.6', '0.7', '0.8', '0.9', '1_spurious']

sweep_configuration = {
    'method': sweep_method,
    'name': sweep_name,
    'metric': {'goal': 'maximize', 'name': 'student test acc'},
    # CHANGE THESE
    'parameters': {
        # 'epochs': {'values': [1]},
        # 'temp': {'distribution': 'uniform', 'min': 15, 'max': 50}, 
        # 'lr': {'distribution': 'log_uniform', 'min': math.log(0.08), 'max': math.log(0.5)},
        'alpha': {'values': [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}, # For grid search
        # 'alpha': {'distribution': 'uniform', 'min': 0, 'max': 1}, # For bayes search
        # 'spurious_corr': {'distribution': 'uniform', 'min': 0, 'max': 1}
    },
    # 'early_terminate': {'type': 'hyperband', 'min_iter': 5}
}

# Dataloaders - regardless of experiment type, always evaluate on these three
plain_test_loader = get_dataloader(load_type ='test', spurious_type='plain', spurious_corr=1, randomize_loc=False)
box_test_loader = get_dataloader(load_type ='test', spurious_type='box', spurious_corr=1, randomize_loc=False)
randbox_test_loader = get_dataloader(load_type ='test', spurious_type='box', spurious_corr=1, randomize_loc=True)

#================================================================================
#================================================================================

# Student model setup (change only if adding to dicts above)
match STUDENT_NUM:
    case 0:
        student = LeNet5(10).to(device)
# # Get names of all submodules in model
# get_submodules(student)

# Teacher model setup (change only if adding to dicts above)
teacher_name = teacher_dict[TEACH_NUM]
load_path = "Image_Experiments/teacher_"+teacher_name
match TEACH_NUM:
    case 0:
        teacher = LeNet5(10).to(device)
    case 1:
        teacher = ResNet50_CIFAR10().to(device)
    case 2:
        teacher = ResNet18_CIFAR10().to(device)
# Load saved teacher model (change only if changing file locations)
load_name = "Image_Experiments/teacher_"+teacher_name+"_"+exp_dict[EXP_NUM]
checkpoint = torch.load(load_name, map_location=device)
teacher.load_state_dict(checkpoint['model_state_dict'])
    
project = exp_dict[EXP_NUM]+"_"+teacher_name+"_"+student_dict[STUDENT_NUM] + "_" + loss_dict[LOSS_NUM]

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
                "spurious type": exp_dict[EXP_NUM],
            }   
        )

    randomize_loc = False
    spurious_corr = 1.0
    name = exp_dict[EXP_NUM]    # Dataset saved name - see dict above
    match EXP_NUM:
        case 0:
            spurious_type = 'plain'
        case 1:
            spurious_type = 'box'
        case 2: 
            spurious_type = 'box'
            randomize_loc = True
        case 3:
            spurious_type = 'box'
            spurious_corr = 0.5
        case 4:
            spurious_type = 'box'
            randomize_loc = True
            spurious_corr = 0.5

    train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)
    test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc)

    # Train
    train_distill(teacher, student, train_loader, test_loader, plain_test_loader, box_test_loader, randbox_test_loader, lr, final_lr, temp, epochs, 1, LOSS_NUM, alpha=alpha)