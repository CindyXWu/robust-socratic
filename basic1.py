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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

NUM_FEATURES = 3
NUM_SIMPLE = 1
COMPLEX = [3, 5]    # Array defining number of slabs for each complex
NOISE = 0
NUM_POINTS = 100

X_train, y_train, X_test, y_test = my_dataloader(gen=True, simple=NUM_SIMPLE, complex=COMPLEX, noise=NOISE, numpoints=NUM_POINTS)
big_model = linear_net(NUM_FEATURES).to_device()

# Loss function
loss_fn = nn.CrossEntropyLoss()
# Create optimizer
optimizer = Adam(big_model.parameters(), lr=lr)
epoch = 20
for epoch in range(epochs):
    for features, labels in tqdm(train_loader):
        scores = big_model(features)
        loss = loss_fn(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        if it % 100 == 0:
            train_acc.append(evaluate(net, train_loader, max_ex=100))
            val_acc.append(evaluate(net, val_loader))
            plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
            plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)
        it += 1
#perform last book keeping
train_acc.append(evaluate(net, train_loader, max_ex=100))
val_acc.append(evaluate(net, val_loader))
plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
plot_acc(train_acc, val_acc, it, it_per_epoch, base_name=output_dir + "acc_"+title, title=title)

