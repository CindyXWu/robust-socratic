import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import os
import wandb
from tqdm import tqdm

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *
from utils_ekdeep import *

# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

output_dir = "Image_Experiments/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")

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

def base_distill_loss(scores, targets, T=1):
    soft_pred = scores/T
    soft_targets = targets/T
    distill_loss = T**2 * ce_loss(soft_pred, soft_targets)
    return distill_loss

# Instantiate losses
kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='batchmean')

def train_teacher(model, train_loader, test_loader, lr, t_epochs):
    """Fine tune a pre-trained teacher model for specific downstream task, or train from scratch."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    it = 0

    for epoch in range(t_epochs):
        print("Epoch: ", epoch)
        train_acc = [1/10]
        test_acc = [1/10]
        train_loss = [0]
        
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores = model(inputs)

            optimizer.zero_grad()
            loss = ce_loss(scores, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc.append(evaluate(model, train_loader, batch_size, max_ex=100))
                test_acc.append(evaluate(model, test_loader, batch_size))
                print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                wandb.log({"teacher test acc": test_acc[-1], "teacher loss": train_loss[-1]})
            it += 1

def train_distill(loss, teacher, student, train_loader, test_loader, lr, final_lr, temp, epochs, repeats):
    """Train student model with distillation loss.
    
    Includes LR scheduling. Change loss function as required. 
    N.B. I need to refator this at some point.
    """
    optimizer = optim.SGD(student.parameters(), lr=lr)
    scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))
    student = student.to(device)

    for _ in range(repeats):
        it = 0
        train_acc = []
        test_acc = []
        train_loss = [0]  # loss at iteration 0
        weight_reset(student)

        for epoch in range(epochs):
            # Student
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)
                scores = student(inputs)
                targets = teacher(inputs)

                # s_map = feature_extractor(student, inputs, batch_size, 2)
                # t_map = feature_extractor(teacher, inputs, batch_size, 2)
                
                ## Jacobian loss
                # input_dim = 32*32*3
                # output_dim = scores.shape[1]
                # loss = jacobian_loss(scores, targets, inputs, 1, 0, batch_size, kl_loss, input_dim, output_dim)
                loss = base_distill_loss(scores, targets, temp)
                ## Feature map loss
                # loss = feature_map_diff(s_map, t_map, False)
                ## Attention jacobian loss
                # loss = jacobian_attention_loss(student, teacher, scores, targets, inputs, batch_size, 1, 0.8, kl_loss)

                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                scheduler.step()
                lr = scheduler.get_lr()
                train_loss.append(loss.detach().cpu().numpy())
                wandb.log({"student loss per iter": train_loss[-1]})

                if it % 100 == 0:
                    batch_size = inputs.shape[0]
                    train_acc.append(evaluate(student, train_loader, batch_size, max_ex=100))
                    test_acc.append(evaluate(student, test_loader, batch_size))
                    # plot_loss(train_loss, it, it_per_epoch, base_name=output_dir+"loss_"+title, title=title)
                    # plot_acc(train_acc, test_acc, it, base_name=output_dir+"acc_"+title, title=title)
                    print('Iteration: %i, %.2f%%' % (it, test_acc[-1]), "Epoch: ", epoch)
                    print("Project {}, LR {}, temp {}".format(project, lr, temp))
                    wandb.log({"student train acc": train_acc[-1], "student test acc": test_acc[-1], "student loss": train_loss[-1]})
                it += 1
        # Logging for sweeps
        wandb.log("student final test acc", test_acc[-1])
        # # Perform last book keeping - only needed for manual plotting
        # train_acc.append(evaluate(student, train_loader, max_ex=100))
        # test_acc.append(evaluate(student, test_loader))

def sweep():
    """Main function for sweep."""
    wandb.init(
        # set the wandb project where this run will be logged
        project=project,
        # track hyperparameters and run metadata
        config={
            "dataset": "CIFAR-100",
            "epochs": epochs,
            "teacher epochs": t_epochs,
            "batch_size": batch_size,
            }
    )

    lr = wandb.config.lr
    temp = wandb.config.temp

    # Models
    resnet = ResNet50_CIFAR10().to(device)
    lenet = LeNet5(10).to(device)
    lenet_to_train = LeNet5(10).to(device)

    # Training-specific variables
    teacher = lenet
    student = lenet
    randomize_loc = False
    spurious_corr = 1.0
    ft_lr = 0.3
    match EXP_NUM:
        case 0:
            spurious_type = 'plain'
            name = 'plain'
        case 1:
            spurious_type = 'box'
            name = 'box'
        case 2:
            spurious_type = 'box'
            name = 'box_random'
            randomize_loc = True
        case 3:
            spurious_type = 'box'
            name = 'box_half'
            spurious_corr = 0.5
        case 4:
            spurious_type = 'box'
            name = 'box_random_half'
            spurious_corr = 0.5
            randomize_loc = True
        case 5:
            teacher = resnet
            spurious_type = 'plain'
            name = 'plain'
            ft_lr = 0.01
        case 6:
            teacher = resnet
            spurious_type = 'box'
            name = 'box'
            ft_lr = 0.01

    # Dataloaders
    train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)
    test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)

    # Fine-tune or train teacher from scratch
    train_teacher(teacher, train_loader, test_loader, ft_lr, t_epochs)

    # Train
    train_distill(jacobian_loss, teacher, student, train_loader, test_loader, lr, temp, epochs, 1)

# CHANGE THESE ==================================================
# Edit this to set wandb to sweep or not
is_sweep = False
# Change this value to change project type and associated hyperparams as appropriate based on dict below
EXP_NUM = 0

exp_dict = {0: 'lenet-lenet', 
            1: 'lenet-lenet-spurious',
            2: 'lenet-lenet-spurious-randomised',
            3: 'lenet-lenet-spurious-0.5',
            4: 'lenet-lenet-spurious-randomised-0.5',
            5: 'resnet-lenet',
            6: 'resnet-lenet-spurious',
            }
project = exp_dict[EXP_NUM]

# Hyperparams - CHANGE THESE
lr = 0.8
ft_lr = 0.5
final_lr = 0.05
temp = 10
epochs = 20
t_epochs = 20
batch_size = 64
dims = [32, 32]

if __name__ == "__main__":

    if is_sweep:
        sweep_configuration = {
            'method': 'grid',
            'name': 'spurious correlation sweep',
            'metric': {'goal': 'maximize', 'name': 'student final test acc',
            },
            # CHANGE THESE
            'parameters': {
                'temp': {'values': [1, 5, 10]}, 
                'lr': {'values': [0.5]},
            }
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=project) 
        wandb.agent(sweep_id, function=sweep, count=20)

    else:
        # Wandb stuff
        wandb.init(
            # set the wandb project where this run will be logged
            project=project,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": "CNN",
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "temp": temp,
                "batch_size": batch_size,
                "teacher": "LeNet5",
                "student": "LeNet5",
                "spurious type": "box",
            }   
        )

    # Models
    resnet = ResNet50_CIFAR10().to(device)
    lenet = LeNet5(10).to(device)
    lenet_to_train = LeNet5(10).to(device)

    # Training-specific variables
    teacher = lenet
    student = lenet
    randomize_loc = False
    spurious_corr = 1.0
    match EXP_NUM:
        case 0:
            spurious_type = 'plain'
            name = 'plain'
        case 1:
            spurious_type = 'box'
            name = 'box'
        case 2:
            spurious_type = 'box'
            name = 'box_random'
            randomize_loc = True
        case 3:
            spurious_type = 'box'
            name = 'box_half'
            spurious_corr = 0.5
        case 4:
            spurious_type = 'box'
            name = 'box_random_half'
            spurious_corr = 0.5
            randomize_loc = True
        case 5:
            teacher = resnet
            spurious_type = 'plain'
            name = 'plain'
            ft_lr = 0.01  
        case 6:
            teacher = resnet
            spurious_type = 'box'
            name = 'box'
            ft_lr = 0.01

    # Dataloaders
    train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)
    test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)

    # Fine-tune or train teacher from scratch
    train_teacher(teacher, train_loader, test_loader, ft_lr, t_epochs)

    # Train
    train_distill(jacobian_loss, teacher, student, train_loader, test_loader, lr, final_lr, temp, epochs, 1)