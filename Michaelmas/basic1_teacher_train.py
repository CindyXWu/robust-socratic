""""Train and test teacher model in model distillation."""
from tqdm import tqdm
import os, sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
from utils import *
from basic1_models import *
from error import *
from plotting import *

# Trying to get rid of 'fail to allocate bitmap' error
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

result_dir = 'Michaelmas/teacher_results/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

@torch.no_grad()
def evaluate(model, dataset, max_ex=0):
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
    return (acc * 100 / ((i+1) * BATCH_SIZE) )

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

# Set GEN to True to generate files from scratch for training and testing
GEN = True
# Number of simple features
NUM_SIMPLE = 1
# Array defining number of slabs for each complex feature
COMPLEX = [5, 8]
# Total number of features
num_features = NUM_SIMPLE + len(COMPLEX)
NUM_POINTS = 3000
# For train
MODE = 1


lr = 0.3
dropout = 0
epochs = 50
BATCH_SIZE = 100
exp =  1

#TRAIN============================================================
loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

old_test_acc = 0
net = linear_net(num_features, dropout=dropout).to(device)

# Fraction of simple datapoints to randomise
fracs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
# List of complex indices (cols) to randomise (see utils.py)
X_list = [[1,2], [1], [2]]
# List of test complex indices (cols) to randomise
SC_list = [[0], [0,1], [0,2]]


# Outer loop to run all experiments
for X in X_list:
    for SC in SC_list:
        # For storing results
        column_names = []
        for frac in fracs:
            column_names.append(f'train_{frac}')
            column_names.append(f'test_{frac}')
        df = pd.DataFrame(columns=column_names)

        X = np.array(X)
        SC = np.array(SC)
        title = f'train_{X}_test_{SC}'
  
        for frac in fracs:
            net.apply(weight_reset)
            net = linear_net(num_features, dropout=dropout).to(device)
            # Train and test dataset names
            FILE_TEST = 'train' + title + str(frac) + '.csv'
            FILE_TRAIN = 'test' + title + str(frac) + '.csv'
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

            optimizer = torch.optim.SGD(net.parameters(), lr=lr)
            optimizer.zero_grad()

            train_acc = []
            test_acc = []
            it_per_epoch = len(train_loader)

            print("Initial train accuracy: ", evaluate(net, train_loader))
            it = 0
            for epoch in range(epochs):
                for features, labels in tqdm(train_loader):
                    out = net(features)
                    scores = sigmoid(out)
                    loss = loss_fn(scores, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if it % 100 == 0:
                        train_acc.append(evaluate(net, train_loader, max_ex=10))
                        test_acc.append(evaluate(net, test_loader, max_ex=10))
                        print("Iteration: ", it, "Train accuracy: ", train_acc[-1], "Test accuracy: ", test_acc[-1])
                    it += 1
            train_acc.append(evaluate(net, train_loader, max_ex=100))
            test_acc.append(evaluate(net, test_loader))
            data = {f'train_{frac}': train_acc, f'test_{frac}': test_acc}
            # Create a temporary DataFrame and append it to the main DataFrame
            df_temp = pd.DataFrame(data)
            df = pd.concat([df, df_temp])

            # Save every model trained on different training data
            # Means teacher models need to be retrained if fracs change for student model
            # Note save name includes X, SC and frac
            torch.save({'model': net,
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f'{result_dir}{title}_{frac}')

        exp += 1

        df.to_csv(f'{result_dir}{title}.csv', index=True)
        plot_df(df, base_name=result_dir, title=title)