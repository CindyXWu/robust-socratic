"""Train and test student model in model distillation"""
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib
import torch.nn.functional as F
from utils import *
from basic1_models import *
from error import *
from plotting import *


# Trying to get rid of 'fail to allocate bitmap' error
matplotlib.use("Agg")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device") 
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))


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
# For train
MODE = 1

# Hyperparameters
lr = 0.3
dropout = 0
epochs = 20
temps = [0, 1]
# For weighted average of scores
alpha = 0.5
sweep = "frac" #"lr", "temp"
exp =  1

def my_loss(scores, targets, T=5):
    """KL Divergence loss with temperature."""
    soft_pred = F.log_softmax(scores/T, dim=1)
    soft_targets = F.softmax(targets/T, dim=1)
    loss = F.kl_div(soft_pred, soft_targets, reduction='batchmean')
    return loss

train_results_df = pd.DataFrame(columns=['X', 'SC', 'temp', 'frac', 'avg_accuracy'])
test_results_df = pd.DataFrame(columns=['X', 'SC', 'temp', 'frac', 'avg_accuracy'])

# Fraction of simple datapoints to randomise
fracs = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
# # List of complex indices (cols) to do split randomise on (see utils.py)
# X_list = [[1, 2], [1], [2]]
# # For test - start with randomising simple feature (first row)
# SC_list = [[0], [0, 1], [0, 2]]
# List of complex indices (cols) to do split randomise on (see utils.py)
X_list = [[1, 2]]
# For test - start with randomising simple feature (first row)
SC_list = [[0]]

#TRAIN=====================================================
# decay_iter = (NUM_POINTS//BATCH_SIZE) * epochs
# cosine_lr_schedule = end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * np.arange(decay_iter)/decay_iter))
sigmoid = nn.Sigmoid()
mseloss_fn = nn.MSELoss()
bceloss_fn = nn.BCELoss()
klloss_fn = nn.KLDivLoss(reduction='batchmean')
models = {}
repeats = 1

for X in X_list:
    for SC in SC_list:
        X = np.array(X)
        SC = np.array(SC)
        
        # For storing results
        column_names = []
        for frac in fracs:
            column_names.append(f'train_{frac}')
            column_names.append(f'test_{frac}')
        df = pd.DataFrame(columns=column_names)

        load_path = "Michaelmas/teacher_results/"
        big_model = linear_net(NUM_FEATURES).to(device)

        output_dir = "Michaelmas/large_student_results/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for temp in temps:
            for frac in fracs:

                loadname = f'train_{X}_test_{SC}_{frac}'
                # Title for saving and plotting
                title = f'train_{X}_test_{SC}_{temp}'

                # Set file names for loading data
                FILE_TEST = "exp" + str(exp) + "test 1 " + str(frac) + ".csv"
                FILE_TRAIN = "exp" + str(exp) + "train 1 " + str(frac) + ".csv"

                checkpoint = torch.load(load_path + loadname, map_location=device)
                big_model.load_state_dict(checkpoint['model_state_dict'])
                big_model.eval()

                small_model = small_linear_net(NUM_FEATURES).to(device)
                small_model.apply(weight_reset)
                small_model.train()

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
                
                optimizer = torch.optim.SGD(small_model.parameters(), lr=lr)
                optimizer.zero_grad()
                it = 0
                train_acc = []
                test_acc = []

                for epoch in range(epochs):
                    for features, labels in tqdm(train_loader):
                        scores = small_model(features)
                        targets = big_model(features)
                        # check scores and targets are in range (0,1)

                        loss = my_loss(scores, targets, T=temp)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
    
                        if it % 100 == 0:
                            train_acc.append(evaluate(small_model, train_loader, max_ex=100))
                            test_acc.append(evaluate(small_model, test_loader))
                            print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                        it += 1
                train_acc.append(evaluate(small_model, train_loader, max_ex=100))
                test_acc.append(evaluate(small_model, test_loader))

                data = {f'train_{frac}': train_acc, f'test_{frac}': test_acc}
                # Create a temporary DataFrame and append it to the main DataFrame
                df_temp = pd.DataFrame(data)
                df = pd.concat([df, df_temp])

                # Only output stats to console of last model
                print(test_acc[-1])
                models[title] = {'model': small_model,
                                    'model_state_dict': small_model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'lr': lr,
                                    'simple frac random': frac,
                                    'train_acc': train_acc,
                                    'test_acc': test_acc,
                                    'iterations': it}
                                    
            df.to_csv(f'{output_dir}{title}.csv', index=True)
            plot_df(df, base_name=output_dir, title=title)

        for key in models.keys():
            print("frac randomised: %s, test_acc: %s" % (models[key]['simple frac random'], models[key]['test_acc']))
            # Note save name includes X, SC and frac
            torch.save({'epoch': epoch,
                    'model_state_dict': models[key]['model_state_dict'],
                    'optimizer_state_dict': optimizer.state_dict()},
                    f'{output_dir}{title}_{frac}')
            
        # test_accs = [models[key]['test_acc'] for key in models.keys()]
        # if sweep == "frac": xs = [models[key]['simple frac random'] for key in models.keys()]

        exp += 1


train_results_df.to_csv(output_dir + "train_avg.csv", index=False)
test_results_df.to_csv(output_dir + "test_avg.csv", index=False)