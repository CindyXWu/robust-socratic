"""Train and test student model in model distillation"""
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
import torch.nn.functional as F
from itertools import product
from utils import *
from basic1_models import *
from error import *
from plotting import *


# Trying to get rid of 'fail to allocate bitmap' error
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

sigmoid = nn.Sigmoid()
mseloss_fn = nn.MSELoss()
bceloss_fn = nn.BCELoss()
klloss_fn = nn.KLDivLoss(reduction='batchmean')

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


# DO NOT change unless very good reason
GEN = False

# Number of simple features
NUM_SIMPLE = 1
# Array defining number of slabs for each complex feature
COMPLEX = [5, 8]
# Total number of features
NUM_FEATURES = NUM_SIMPLE + len(COMPLEX)
NUM_POINTS = 3000
BATCH_SIZE = 50
MODE = 1 # train

# Hyperparameters
lr = 0.3
dropout = 0
epochs = 50
temp = 1
# For weighted average of scores
alpha = 0.5
sweep = "frac" #"lr", "temp"
exp =  1
repeats = 1

# Custom distillation loss function to match sigmoid output from teacher to student with MSE loss
def my_loss(scores, targets, T=5):
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    loss = bceloss_fn(soft_pred, soft_targets)
    return loss

train_results_df = pd.DataFrame(columns=['X', 'SC', 'temp', 'frac', 'avg_accuracy'])
test_results_df = pd.DataFrame(columns=['X', 'SC', 'temp', 'frac', 'avg_accuracy'])


# List of complex indices (cols) to do split randomise on (see utils.py)
X_list = [[1, 2], [1], [2]]
# For test - start with randomising simple feature (first row)
SC_list = [[0], [0, 1], [0, 2]]
# # List of complex indices (cols) to do split randomise on (see utils.py)
# X_list = [[1, 2]]
# # For test - start with randomising simple feature (first row)
# SC_list = [[0]]
# Fraction of simple datapoints to randomise
fracs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]

for x_i, X in enumerate(X_list):
    for sc_i, SC in enumerate(SC_list):
        X = np.array(X)
        SC = np.array(SC)
        title = f'train_{X}_test_{SC}'

        # For storing results
        column_names = []
        for frac in fracs:
            column_names.append(f'train_{frac}')
            column_names.append(f'test_{frac}')
        df = pd.DataFrame(columns=column_names)

        load_path = "Michaelmas/teacher_results/"
        teacher = linear_net(NUM_FEATURES).to(device)

        output_dir = "Michaelmas/large_student_results/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for frac in fracs:
            loadname = f'train_{X}_test_{SC}_{frac}'

            # Set file names for loading data
            FILE_TEST = 'train' + title + str(frac) + '.csv'
            FILE_TRAIN = 'test' + title + str(frac) + '.csv'

            checkpoint = torch.load(load_path + loadname, map_location=device)
            teacher.load_state_dict(checkpoint['model_state_dict'])
            teacher.eval()

            student = small_linear_net(NUM_FEATURES).to(device)
            student.apply(weight_reset)
            student.train()

            X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, frac=frac, x=X)

            # Reshape y tensor to (datapoints*1)
            y_train = y_train.reshape(-1,1)
            train_dataset = CustomDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            it_per_epoch = len(train_loader)

            # Test dataset has same number of points as train
            X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, sc=SC)
            y_test = y_test.reshape(-1,1)
            test_dataset = CustomDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            optimizer = torch.optim.SGD(student.parameters(), lr=lr)
            optimizer.zero_grad()
            it = 0
            train_acc = []
            test_acc = []

            for epoch in range(epochs):
                for features, labels in tqdm(train_loader):
                    scores = student(features)
                    targets = teacher(features)
                    assert not torch.isnan(targets).any(), "Teacher's output contains NaN values"

                    loss = my_loss(scores, targets, T=temp)
                    optimizer.zero_grad()
                    loss.backward()

                    # # Detecting exploding gradients
                    # for name, parameter in student.named_parameters():
                    #     if parameter.grad is not None:
                    #         assert not torch.isnan(parameter.grad).any(), f"NaN gradient in {name}"
                    #         assert not torch.isinf(parameter.grad).any(), f"Infinite gradient in {name}"
                    # torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

                    optimizer.step()

                    if it % 100 == 0:
                        train_acc.append(evaluate(student, train_loader, max_ex=100))
                        test_acc.append(evaluate(student, test_loader))
                        print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                    it += 1
            
            train_acc.append(evaluate(student, train_loader, max_ex=100))
            test_acc.append(evaluate(student, test_loader))

            # Create a temporary DataFrame and append it to the main DataFrame
            data = {f'train_{frac}': train_acc, f'test_{frac}': test_acc}
            df_temp = pd.DataFrame(data)
            df = pd.concat([df, df_temp])
            print(test_acc[-1])

            # Note save name includes X, SC and frac
            torch.save({'model': student,
                    'epoch': epoch,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                    f'{output_dir}{title}_{frac}')

        exp += 1

        print(df.head())                      
        df.to_csv(f'{output_dir}{title}.csv', index=True)
        plot_df(df, base_name=output_dir, title=title)



train_results_df.to_csv(output_dir + "train_avg.csv", index=False)
test_results_df.to_csv(output_dir + "test_avg.csv", index=False)