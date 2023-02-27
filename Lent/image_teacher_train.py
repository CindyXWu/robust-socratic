"""Separate file to train teacher in case of need for hyperparameter tuning. Also saves teacher model."""
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
# Change directory to one this file is in
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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

def train_teacher(model, train_loader, test_loader, lr, final_lr, epochs, save=False):
    """Fine tune a pre-trained teacher model for specific downstream task, or train from scratch."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    it = 0
    scheduler = LR_Scheduler(optimizer, epochs, base_lr=lr, final_lr=final_lr, iter_per_epoch=len(train_loader))

    for epoch in range(epochs):
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
            scheduler.step()
            lr = scheduler.get_lr()
            train_loss.append(loss.detach().cpu().numpy())
            
            if it % 100 == 0:
                batch_size = inputs.shape[0]
                train_acc.append(evaluate(model, train_loader, batch_size, max_ex=100))
                test_acc.append(evaluate(model, test_loader, batch_size))
                print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                wandb.log({"teacher test acc": test_acc[-1], "teacher loss": train_loss[-1], "teacher lr": lr})
            it += 1

    if save:
        torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss},
                output_dir + "teacher_"+ teacher_dict[TEACH_NUM])
        
    # Log final test accuracy for sweep metrics
    wandb.log({"teacher final test acc": test_acc[-1]})

def base_distill_loss(scores, targets, T=1):
    soft_pred = scores/T
    soft_targets = targets/T
    distill_loss = T**2 * ce_loss(soft_pred, soft_targets)
    return distill_loss

# Instantiate losses
kl_loss = nn.KLDivLoss(reduction='batchmean')
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = nn.MSELoss(reduction='batchmean')

def sweep_teacher():
    wandb.init(
        # set the wandb project where this run will be logged
        project=teacher_name,
        # track hyperparameters and run metadata
        config={
            "teacher": teacher_dict[TEACH_NUM],
            "dataset": "CIFAR-100",
            "batch_size": batch_size,
            }
    )

    lr = wandb.config.lr
    final_lr = wandb.config.final_lr
    epochs = wandb.config.epochs

    name = exp_dict[EXP_NUM]
    randomize_loc = False
    spurious_corr = 1
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
    train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)
    test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)

    # Fine-tune or train teacher from scratch
    train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs)

# Change this value as appropriate based on dict below
TEACH_NUM = 0
teacher_dict = {0: "LeNet5_CIFAR10", 
                1: "ResNet50_CIFAR10"}
teacher_name = teacher_dict[TEACH_NUM]
match TEACH_NUM:
    case 0:
        teacher = LeNet5(10).to(device)
    case 1:
        teacher = ResNet50_CIFAR10().to(device)
print("Teacher: ", teacher_dict[TEACH_NUM])

# Change this to change which dataset the teacher is trained on =============================================
EXP_NUM = 0
exp_dict = {0: 'plain', 1: 'box', 2: 'box_random', 3: 'box_half', 4: 'box_random_half'}

# Hyperparams - CHANGE THESE ================================================================================
lr = 0.5
final_lr = 0.1
epochs = 20
batch_size = 64
dims = [32, 32]
sweep_count = 4

is_sweep = True
if __name__ == "__main__":
    if is_sweep == True:
        sweep_configuration = {
            'method': 'bayes',
            'name': teacher_name,
            'metric': {'goal': 'maximize', 'name': 'teacher final test acc',
            },
            # CHANGE THESE ==============================================================
            'parameters': {
                'epochs': {'values': [15, 25, 35]},
                'lr': {'values': [1, 0.5, 0.3]},
                'final_lr': {'values': [0.3, 0.1, 0.05]}
            },
            'early_terminate': {'type': 'hyperband', 'min_iter': 500}
        }
        sweep_id = wandb.sweep(sweep=sweep_configuration, project=teacher_name) 
        wandb.agent(sweep_id, function=sweep_teacher, count=sweep_count)
    # Should be used for retraining once best model indentified from sweep
    else:
    # Save teacher model after run
        # Wandb stuff
        wandb.init(
            # set the wandb project where this run will be logged
            project=teacher_name,
            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "dataset": "CIFAR-10",
                "epochs": epochs,
                "batch_size": batch_size,
                "spurious type": exp_dict[EXP_NUM]
            }   
        )

        name = exp_dict[EXP_NUM]
        randomize_loc = False
        spurious_corr = 1
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
        train_loader = get_dataloader(load_type='train', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)
        test_loader = get_dataloader(load_type ='test', spurious_type=spurious_type, spurious_corr=spurious_corr, randomize_loc=randomize_loc, name=name)

        # Fine-tune or train teacher from scratch
        train_teacher(teacher, train_loader, test_loader, lr, final_lr, epochs, save=True)
