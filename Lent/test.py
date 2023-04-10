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
device = "cuda" if torch.cuda.is_available() else "cpu"
load_name = "Lent/Image_Experiments/teacher_"+"ResNet18_CIFAR100_BR"
checkpoint = torch.load(load_name, map_location=device)

for key in checkpoint.keys():
    print(key)

print(checkpoint['loss_hist'][-1])