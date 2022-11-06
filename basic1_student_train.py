"""Train and test student model in model distillation"""

from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from utils import *
from basic1_models import *
from error import *
from plotting import *
import matplotlib

# Trying to get rid of 'fail to allocate bitmap' error
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# Load teacher network
load_path = "teacher_linear_model/"
big_model = linear_net().to(device)
checkpoint = torch.load(load_path + "teacher saved")
big_model.load_state_dict(checkpoint['model_state_dict'])
big_model.eval()

GEN = False
# Train and test dataset names
FILE_TEST = "test 1.csv"
FILE_TRAIN = "train 1.csv"

# Number of simple features
NUM_SIMPLE = 1
# Array defining number of slabs for each complex feature
COMPLEX = [5, 8]
# Total number of complex features
num_features = NUM_SIMPLE + len(COMPLEX)
NUM_POINTS = 10000
BATCH_SIZE = 500
# For train
MODE = 1
# Fraction of simple datapoints to randomise
fracs = [0.4, 0.6, 1]
X = [1,2]
# For test - start with randomising simple feature (first row)
SC = [0]

# Hyperparameters
lr = 0.5
dropout = 0
epochs = 300
start_lr = 10
end_lr = 0.001

#TRAIN=====================================================
loss_fn = nn.BCELoss()
decay_iter = (NUM_POINTS//BATCH_SIZE) * epochs
cosine_lr_schedule = end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * np.arange(decay_iter)/decay_iter))