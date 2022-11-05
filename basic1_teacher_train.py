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
1
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# Edited a bunch ===============================================================
# Set GEN to True to generate files from scratch for training and testing
# Make sure to do this whenever you change the number of points or randomisation (section below)
GEN = True
# Train and test dataset names
FILE_TEST = "test 1.csv"
FILE_TRAIN = "train 1.csv"

# Edited sometimes ============================================
# Number of simple features
NUM_SIMPLE = 1
# Array defining number of slabs for each complex feature
COMPLEX = [5, 8]
# Total number of complex features
num_features = NUM_SIMPLE + len(COMPLEX)
NUM_POINTS = 10000
BATCH_SIZE = 100
# For train
MODE = 1
# Fraction of simple datapoints to randomise
fracs = [0, 0.2, 0.4, 0.6, 0.8, 1]
X = [1,2]
# For test - start with randomising simple feature (first row)
SC = [0]

# Hyperparameters
lr = 5e-4
dropout = 0.4
epochs = 50

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        print('data shape: ', self.features.shape)
        print('labels shape: ', self.labels.shape)

    def __getitem__(self, index):
        # None adds extra dimension which hopefully forces dataloader to load correct size
        f = torch.tensor(self.features[index, :])
        l = torch.tensor(self.labels[index])
        return (f.to(device), l.to(device))

    def __len__(self):
        return len(self.labels)

def evaluate(model, dataset, max_ex=0):
    acc = 0
    for i, (features, labels) in enumerate(dataset):
        # Batch size in length, varying from 0 to 1
        scores = model(features)
        # Predictive class is closest from sigmoid output
        pred = torch.round(scores, decimals=0)
        # Save to pred 
        acc += torch.sum(torch.eq(pred, labels)).item()
        if max_ex != 0 and i >= max_ex:
            break
    # Return average accuracy as a percentage
    # Fraction of data points correctly classified
    return (acc * 100 / ((i+1) * BATCH_SIZE) )

# Set output directory and create if needed
output_dir = "teacher_linear_model/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#TRAIN============================================================
loss_fn = nn.MSELoss()
models = {}
old_test_acc = 0
for frac in fracs:
    #DATA STUFF======================================================
    # Train dataset
    X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, frac=frac, x=X)
    # Reshape y tensor tp (datapoints*1)
    y_train = y_train.reshape(-1,1)
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Test dataset has 1/4 number of points
    X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS//4, sc=SC)
    # Reshape y tensor
    y_test = y_test.reshape(-1,1)
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # title = 'lr=' + str(lr)
    title = 'simple randomised=' + str(frac)
    print("\n", title, "\n")
    # Instantiate a new network
    net = linear_net(num_features, dropout=dropout).to(device)
    # Create optimizer
    optimizer = Adam(net.parameters(), lr=lr)
    optimizer.zero_grad()
    # Start training
    train_acc = []
    test_acc = []
    train_loss = [0]  # loss at iteration 0

    # print(len(train_data), len(train_loader))
    it_per_epoch = len(train_loader)
    print("iterations per epoch: ", it_per_epoch)
    # Count number of epochs
    it = 0
    for epoch in range(epochs):
        for features, labels in tqdm(train_loader):
            # Forward pass
            scores = net(features)
            loss = loss_fn(scores, labels)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            # Evaluate model at this iteration
            if it % 100 == 0:
                train_acc.append(evaluate(net, train_loader, max_ex=10))
                test_acc.append(evaluate(net, test_loader))
                plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)
            it += 1
    # Perform last book keeping
    train_acc.append(evaluate(net, train_loader, max_ex=100))
    test_acc.append(evaluate(net, test_loader))
    plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
    plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)

    # Save model to dictionary, titled by dropout (can be changed)
    models[title] = {'model': net,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_hist': train_loss,
                        'lr':lr,
                        'p':dropout,
                        'test_acc': test_acc[-1]}

for key in models.keys():
    print("for lr: %s, test_acc: %s" % (models[key]['lr'], models[key]['test_acc']))
    # print(key)

test_accs = [models[key]['test_acc'] for key in models.keys()]
xs = [models[key]['p'] for key in models.keys()]
keys = [key for key in models.keys()]

best_key = keys[np.argmax(test_accs)]
print(best_key)
best_model = models[best_key]['model']

# torch.save({'epoch': epoch,
#             'model_state_dict': best_model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss_hist': train_loss},
#             output_dir + "main teacher model")