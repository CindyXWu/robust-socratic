import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import torchvision.models as models
from torch import nn, optim
import os
from image_models import *
from losses import *
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

def load_cifar_10(dims):
    """Load CIFAR-10 dataset and return dataloaders.
    :param dims: tuple, dimensions of the images
    """
    transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,  
                                0.406], [0.229, 0.224, 0.225])])
    trainset = datasets.CIFAR10(root='./data', download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    return trainset, testset, trainloader, testloader

# ResNet50 setup
resnet = models.resnet50(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 10)
resnet = resnet.to(device)

def weight_reset(m):
    """Reset weights of model at start of training."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Training
def train_distill(loss, teacher, student, lr, epochs, trainloader, testloader):
    """Train student model with distillation loss."""
    optimiser = optim.SGD(student.parameters(), lr=lr)
