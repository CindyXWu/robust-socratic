""""Train and test teacher model in model distillation."""

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

GEN = True
FILE_TEST = "Test 1.csv"
FILE_TRAIN = "Train 1.csv"
NUM_SIMPLE = 1
COMPLEX = [3, 5]    # Array defining number of slabs for each complex
num_features = NUM_SIMPLE + len(COMPLEX)
NUM_POINTS = 100
BATCH_SIZE = 10
# For train
MODE = 1
X = [1,2]
# For test - start with randomising simple feature (first row)
SC = [0]

# Hyperparameters
lrs = [5e-4]
dropouts = [0.4]
epochs = 10

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        print('data shape: ', self.features.shape)
        print('labels shape: ', self.labels.shape)

    def __getitem__(self, index):
        f = torch.tensor(self.features[:,index])
        l = torch.tensor(self.labels[index])
        return (f.to(device), l.to(device))

    def __len__(self):
        return len(self.labels)

def evaluate(model, dataset, max_ex=0):
    acc = 0
    N = len(dataset) * BATCH_SIZE
    for i, (features, labels) in enumerate(dataset):
        # Batch size in length, varying from 0 to 1
        scores = model(features)
        # Predictive class is closest from sigmoid output
        pred = np.round(scores, 0)
        # Save to pred 
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # print(i)
    return (acc * 100 / ((i+1) * BATCH_SIZE) )

#DATA STUFF======================================================
# Train dataset
X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, x=X)
train_dataset = CustomDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test dataset has 1/4 number of points
X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS//4, sc=SC)
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set output directory and create if needed
output_dir = "teacher_linear_model/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("training data", X_train)
print(y_train)
# #TRAIN============================================================
# loss_fn = nn.MSELoss()
# models = {}
# for lr in lrs:
#     for dropout in dropouts:
#         # title = 'lr=' + str(lr)
#         title = 'dropout p=' + str(dropout)
#         print("\n", title, "\n")
#         # Instantiate a new network
#         net = linear_net(num_features, dropout=dropout).to(device)
#         # Create optimizer
#         optimizer = Adam(net.parameters(), lr=lr)
#         optimizer.zero_grad()
#         # Start training
#         val_acc = []
#         train_acc = []
#         train_loss = []  # loss at iteration 0

#         # print(len(train_data), len(train_loader))
#         it_per_epoch = len(train_loader)
#         it = 0
#         for epoch in range(epochs):
#             for features, labels in tqdm(train_loader):
#                 scores = net(features)
#                 loss = loss_fn(scores, labels)
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 if it % 10 == 0:
#                     train_acc.append(evaluate(net, train_loader, max_ex=10))
#                     plot_acc(train_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
#                 it += 1
#         #perform last book keeping
#         train_acc.append(evaluate(net, train_loader, max_ex=100))
#         plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
#         plot_acc(train_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
#         models[title] = {'model': net,
#                          'model_state_dict': net.state_dict(),
#                          'optimizer_state_dict': optimizer.state_dict(),
#                          'loss_hist': train_loss,
#                          'lr':lr,
#                          'p':dropout,
#                          'val_acc': val_acc[-1]}