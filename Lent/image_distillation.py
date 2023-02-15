import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch import nn, optim
import os
from tqdm import tqdm

from image_models import *
from plotting import *
from jacobian_srinivas import *
from contrastive import *
from feature_match import *

# Setup ========================================================================
# Suppress warnings "divide by zero" produced by NaN gradients
import warnings
warnings.filterwarnings("ignore")

output_dir = "Image Experiments/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Hyperparams ========================================================================
lr = 0.01
dropout = 0
temps = [1, 5]
alphas = [0.25, 0.5, 0.75]
# Use a long distillation training schedule
epochs = 3
t_epochs = 100
BATCH_SIZE = 64

# Define loss functions
bceloss_fn = nn.BCELoss()
kldivloss = nn.KLDivLoss(reduction='batchmean')

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, drop_last=True)
    return trainset, testset, trainloader, testloader

def evaluate(model, dataset, max_ex=0):
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
    return (acc*100 / ((i+1)*BATCH_SIZE))

def weight_reset(model):
    """Reset weights of model at start of training."""
    for layer in model.modules():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def fine_tune(model, dataloader, title):
    """Fine tune a pre-trained teacher model for specific downstream task."""
    optimizer = optim.SGD(model.parameters(), lr=lr)
    it - 0
    it_per_epoch = len(dataloader)
    
    for _ in range(t_epochs):
        train_acc = []
        test_acc = []
        train_loss = [0]
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = bceloss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            if it % 100 == 0:
                    train_acc.append(evaluate(model, train_loader, max_ex=100))
                    test_acc.append(evaluate(model, test_loader))
                    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                    plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)
                    print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
            it += 1
    
def train_distill(loss, teacher, student, lr, epochs, repeats, title):
    """Train student model with distillation loss."""
    optimizer = optim.SGD(student.parameters(), lr=lr)
    student = student.to(device)
    for _ in range(repeats):
        it = 0
        train_acc = []
        test_acc = []
        train_loss = [0]  # loss at iteration 0
        it_per_epoch = len(train_loader)
        for _ in range(epochs):
            weight_reset(student)
            # Student
            for inputs, labels in tqdm(train_loader):
                inputs = inputs.to(device)
                inputs.requires_grad = True
                labels = labels.to(device)
                # Student outputs
                scores = student(inputs)
                # Teacher outputs
                targets = teacher(inputs)
                input_dim = 32*32*3
                output_dim = scores.shape[1]
                loss = jacobian_loss(scores, targets, inputs, 1, 0.8, BATCH_SIZE, input_dim, output_dim, bceloss_fn)
                loss.backward()
                optimizer.zero_grad()
                optimizer.step()
                train_loss.append(loss.item())
                
                if it % 100 == 0:
                    train_acc.append(evaluate(student, train_loader, max_ex=100))
                    test_acc.append(evaluate(student, test_loader))
                    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                    plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)
                    print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                it += 1

        # Perform last book keeping
        train_acc.append(evaluate(student, train_loader, max_ex=100))
        test_acc.append(evaluate(student, test_loader))
        plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
        plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)

if __name__ == "__main__":
    # Get data
    train_set, test_set, train_loader, test_loader = load_cifar_10((32, 32))
    lenet = LeNet5(10)
    # ResNet50 early layers modified for CIFAR-10
    resnet = ResNet50_CIFAR10()
    for param in resnet.parameters():
        param.requires_grad = False
    resnet = resnet.to(device)
    # Train model
    train_distill(jacobian_loss, resnet, lenet, lr, epochs, 1, "lenet_jac")