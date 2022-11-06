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
import matplotlib

# Trying to get rid of 'fail to allocate bitmap' error
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

# Set GEN to True to generate files from scratch for training and testing
GEN = True
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
fracs = [0.4, 1]
X = [1,2]
# For test - start with randomising simple feature (first row)
SC = [0]

# Hyperparameters
lr = 0.5
dropout = 0
epochs = 200

# Set output directory and create if needed
output_dir = "teacher_linear_model/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to evaluate model accuracy
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

#TRAIN============================================================
loss_fn = nn.BCELoss()
models = {}
old_test_acc = 0
for frac in fracs:
    # Train dataset
    X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, frac=frac, x=X)
    # Reshape y tensor tp (datapoints*1)
    y_train = y_train.reshape(-1,1)
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Test dataset has same number of points as train
    X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, sc=SC)
    # Reshape y tensor
    y_test = y_test.reshape(-1,1)
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    title = 'Fraction randomised=' + str(frac)
    print("\n", title, "\n")
    # Instantiate a new network
    net = linear_net(num_features, dropout=dropout).to(device)
    # Create optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
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
                # max_ex is the number of batches to evaluate
                train_acc.append(evaluate(net, train_loader, max_ex=10))
                test_acc.append(evaluate(net, test_loader, max_ex=10))
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
                        'simple frac random':frac,
                        'lr':lr,
                        'test_acc': test_acc[-1]}

for key in models.keys():
    print("frac randomised: %s, test_acc: %s" % (models[key]['simple frac random'], models[key]['test_acc']))

test_accs = [models[key]['test_acc'] for key in models.keys()]
xs = [models[key]['simple frac random'] for key in models.keys()]
keys = [key for key in models.keys()]
print(keys)

# Plot summary
fig = plt.figure(figsize=(8, 4), dpi=100)
plt.scatter(xs, test_accs)
plt.title("{0} Epochs".format(epochs))
plt.ylabel('Test accuracy')
plt.xlabel('Randomised fraction of simple feature during training')
# plt.xscale('log')
# plt.xlim([9e-5, 5e-1])
fig.savefig(output_dir + 'summary_{0}epochs.png'.format(epochs))

# Save model that's been trained on 100% randomised simple features
# This is bad code and might be broken if the input to frac changes
last_model = models[keys[-1]]['model']

torch.save({'epoch': epoch,
            'model_state_dict': last_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_hist': train_loss},
            output_dir + "teacher saved")