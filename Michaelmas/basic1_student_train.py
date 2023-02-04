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
# Fraction of simple datapoints to randomise
fracs = [0, 1, 0.5, 0.1]
# List of complex indices (cols) to do split randomise on (see utils.py)
X_list = [[1,2], [1,2], [1,2], [1], [1], [1], [2], [2], [2]]
# For test - start with randomising simple feature (first row)
SC_list = [[0], [0,1], [0,2], [0], [0,1], [0,2], [0], [0,1], [0,2]]

# Hyperparameters
lr = 0.3
dropout = 0
epochs = 150
temps = [5]
# For weighted average of scores
alpha = 0.5
sweep = "frac" #"lr", "temp"
exp =  1

# Custom distillation loss function to match sigmoid output from teacher to student with MSE loss
def my_loss(scores, targets, T=5):
    soft_pred = sigmoid(scores/T)
    soft_targets = sigmoid(targets/T)
    loss = T**2 * bceloss_fn(soft_pred, soft_targets)
    return loss

# Function to evaluate model accuracy
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
    # Return average accuracy as a percentage
    # Fraction of data points correctly classified
    return (acc * 100 / ((i+1) * BATCH_SIZE) )

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


#TRAIN=====================================================
# decay_iter = (NUM_POINTS//BATCH_SIZE) * epochs
# cosine_lr_schedule = end_lr + 0.5 * (start_lr - end_lr) * (1 + np.cos(np.pi * np.arange(decay_iter)/decay_iter))
sigmoid = nn.Sigmoid()
mseloss_fn = nn.MSELoss()
bceloss_fn = nn.BCELoss()

models = {}

repeats = 5
# Outer loop to run all experiments

for X, SC in zip(X_list, SC_list):
    X = np.array(X)
    SC = np.array(SC)
    # Instantiate networks
    load_path = "teacher_linear_" + str(exp) + "/"
    big_model = linear_net(NUM_FEATURES).to(device)
    output_dir = "small_linear_"+str(exp)+"/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lists are length fracs long, and added to train_avg and test_avg as rows later
    exp_train_results = [0]
    exp_test_results = [0]

    for temp in temps:
        for frac in fracs:
            # Set file names
            FILE_TEST = "exp" + str(exp) + "test 1 " + str(frac) + ".csv"
            FILE_TRAIN = "exp" + str(exp) + "train 1 " + str(frac) + ".csv"

            # Load teacher model (fn of frac randomised)
            loadname = "Fraction simple randomised " + str(frac)
            print("\n", loadname, "\n")
            checkpoint = torch.load(load_path + "teacher_" + loadname, map_location=device)
            big_model.load_state_dict(checkpoint['model_state_dict'])
            big_model.eval()

            # Load train data
            X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, frac=frac, x=X)
            # Reshape y tensor to (datapoints*1)
            y_train = y_train.reshape(-1,1)
            train_dataset = CustomDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # # debug data
            # print("Length of dataloader: ", len(train_loader))
            # print("Length of train dataset: ", len(train_dataset))
            # print("Training dataset shape:", train_dataset[0][0].shape, train_dataset[0][1].shape)

            # Test dataset has same number of points as train
            X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, sc=SC)
            # Reshape y tensor
            y_test = y_test.reshape(-1,1)
            test_dataset = CustomDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

            # Title for saving and plotting
            title = "Frac " + str(frac) + " Temp " + str(temp)

            for rep in range(repeats):
                small_model = small_linear_net(NUM_FEATURES).to(device)
                small_model.apply(weight_reset)
                optimizer = torch.optim.SGD(small_model.parameters(), lr=lr)
                optimizer.zero_grad()
                it = 0
                train_acc = []
                test_acc = []
                train_loss = [0]  # loss at iteration 0
                it_per_epoch = len(train_loader)
                for epoch in range(epochs):
                    for features, labels in tqdm(train_loader):
                        # rnd_features = torch.rand(BATCH_SIZE, NUM_FEATURES).to(device)
                        scores = small_model(features)
                        targets = big_model(features)

                        #loss = my_loss(scores, targets, T=temp)
                        loss = bceloss_fn(sigmoid(scores), labels)
                        # if it == 0:
                        #     print(scores.size())
                        #     print(scores[0])
                        #     print(targets[0])
                        #     soft_pred = softmax_op(scores / temp)
                        #     soft_targets = softmax_op(targets / temp)
                        #     print(soft_pred[0])
                        #     print(soft_targets[0])
                        #     loss_rep = mseloss_fn(soft_pred, soft_targets)
                        #     print(loss)
                        #     print(loss_rep)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_loss.append(loss.item())
                        if it % 100 == 0:
                            train_acc.append(evaluate(small_model, train_loader, max_ex=100))
                            test_acc.append(evaluate(small_model, test_loader))
                            plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                            plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)
                            print('Iteration: %i, %.2f%%' % (it, test_acc[-1]))
                        it += 1
                #perform last book keeping
                train_acc.append(evaluate(small_model, train_loader, max_ex=100))
                test_acc.append(evaluate(small_model, test_loader))
                plot_loss(train_loss, it, it_per_epoch, base_name=output_dir + "loss_"+title, title=title)
                plot_acc(train_acc, test_acc, it, base_name=output_dir + "acc_"+title, title=title)

                exp_test_results[-1] += np.mean(test_acc[-5:])
                exp_train_results[-1] += np.mean(train_acc[-5:])
                # Once all repeats are done, divide by repeats to get average
                if rep == repeats - 1:
                    # Calculate average
                    exp_test_results[-1] /= repeats
                    exp_train_results[-1] /= repeats
                    # Add new element to account for next frac
                    exp_test_results.append(0)
                    exp_train_results.append(0)

            # Only output stats to console of last model
            print(test_acc[-1])
            models[title] = {'model': small_model,
                                'model_state_dict': small_model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr': lr,
                                'simple frac random': frac,
                                'loss_hist': train_loss,
                                'train_acc': train_acc,
                                'test_acc': test_acc,
                                'iterations': it}
        
    # Convert to arrays and save
    np.savetxt(output_dir + "train_avg.csv", np.array(exp_train_results), delimiter=",")
    np.savetxt(output_dir + "test_avg.csv", np.array(exp_test_results), delimiter=",")

    for key in models.keys():
        print("frac randomised: %s, test_acc: %s" % (models[key]['simple frac random'], models[key]['test_acc']))

    test_accs = [models[key]['test_acc'] for key in models.keys()]
    if sweep == "frac": xs = [models[key]['simple frac random'] for key in models.keys()]

    # keys = [key for key in models.keys()]
    # best_key = keys[np.argmax(test_accs)]
    # print(best_key)
    # best_model = models[best_key]['model']
    # best_model.eval()

    exp += 1