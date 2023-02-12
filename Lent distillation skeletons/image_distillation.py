import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch import nn, optim
import os
from tqdm import tqdm
from image_models import *
from losses import jacobian, contrastive
from plotting import *
# import time
# import torchvision
# import copy

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
epochs = 1000
BATCH_SIZE = 64

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
    return trainset, testset, trainloader, testloader

# ResNet50 setup
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)

def evaluate(model, dataset, max_ex=0):
    """Evaluate model accuracy on dataset."""
    acc = 0
    for i, (features, labels) in enumerate(dataset):
        # Batch size in length, varying from 0 to 1
        scores = sigmoid(model(features))
        # Predictive class is closest from sigmoid output
        pred = torch.round(scores, decimals=0)
        # Save to pred 
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # Return average accuracy as a percentage
    # Fraction of data points correctly classified
    return (acc*100 / ((i+1)*BATCH_SIZE))

def weight_reset(m):
    """Reset weights of model at start of training."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Get data
train_set, test_set, train_loader, test_loader = load_cifar_10((224, 224))
small_train_set, small_test_set, small_train_loader, small_test_loader = load_cifar_10((32, 32))

# print(train_set[0][0].size())
# print(small_train_set[0][0].size())

def train_distill(loss, teacher, student, lr, epochs, trainloader, testloader, repeats, title, **kwargs):
    """Train student model with distillation loss."""
    optimiser = optim.SGD(student.parameters(), lr=lr)
    for rep in range(repeats):
        student = student.to(device)
        student.apply(weight_reset)
        optimizer = torch.optim.SGD(student.parameters(), lr=lr)
        it = 0
        train_acc = []
        test_acc = []
        train_loss = [0]  # loss at iteration 0
        it_per_epoch = len(train_loader)
        for epoch in range(epochs):
            dataloader_iterator = iter(train_loader)
            for i, (inputs1, labels1) in tqdm(enumerate(small_train_loader), total=len(small_train_loader)):
                try: 
                    inputs2, labels2 = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(train_loader)
                    inputs2, labels2 = next(dataloader_iterator)

                inputs1 = inputs1.to(device)
                inputs1.requires_grad = True
                inputs2 = inputs2.to(device)
                inputs2.requires_grad = True

                scores = student(inputs1)
                targets = teacher(inputs2)

                s_jac = torch.autograd.grad(scores, labels1, grad_outputs=torch.ones_like(scores), create_graph=True)[0]
                print(s_jac.size())
                loss = loss(scores, targets, **kwargs)

                optimizer.zero_grad()
                loss.backward()
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

lenet = LeNet5(10)
train_distill(jacobian_loss, resnet, lenet, lr, epochs, train_loader, test_loader, 1, "lenet_jac", temp=1)