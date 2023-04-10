import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
from collections import defaultdict
from info_dicts import *

api = wandb.Api()

def plot_loss(loss, it, it_per_epoch, smooth_loss=[], base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    # The iteration marking the start of each epoch
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    try:
        loss_for_epochs = [loss[i] for i in epochs]
        plt.plot(epochs, loss_for_epochs, linestyle='', marker='o')
    except IndexError:
        pass
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    if base_name != '':
        fig.savefig(base_name + '.png')
    else:
        plt.show()
    plt.close("all")

def plot_loss(loss, it, it_per_epoch, smooth_loss=[], base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [loss[i] for i in epochs], linestyle='', marker='o')
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    if base_name != '':
        fig.savefig(base_name + '.png')
    else:
        plt.show()
    plt.close("all")

def plot_acc(train_acc, test_acc, it, base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    if it !=0:
        inter = it//(len(train_acc) -1)
        x_axis = [i * inter for i in range(len(train_acc))]
    else:
        x_axis = [0]
    plt.plot(x_axis, train_acc, label="Train")
    plt.plot(x_axis, test_acc, label="Test")
    plt.legend()
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.ylim([20, 110])
    if base_name != '':
        fig.savefig(base_name + '.png')
    else:
        plt.show()
    plt.close("all")

def plot_data(data_arr, d1, d2):
    """"Toy plot function that only plots two features against each other."""
    # Plot scatter colour as red if y = 0 and blue if y = 1
    plt.scatter(data_arr[:, d1], data_arr[:, d2], c=data_arr[:, -1], cmap='bwr')
    plt.xlabel('Feature ' + str(d1))
    plt.ylabel("Feature " + str(d2))
    plt.show()
    
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        break

def wandb_plot_CIFAR100(project_name, t_num, s_num, exp_num):
    runs = api.runs(project_name)
    teacher = 'ResNet18_CIFAR100' #teacher_dict[t_num]
    student = 'ResNet18_Flexi' #student_dict[s_num]
    # grouped_runs = defaultdict(list)
    t_mech = exp_dict[exp_num]
    # Filter
    runs = [run for run in runs if 
            run.summary.get('Teacher') == teacher and 
            run.summary.get('Student') == student and 
            run.summary.get('Teacher mechanism') == t_mech
            ]
    run_histories = []
    for run in runs:
        history = run.history()
        step_values = history['step'].values
        BR_test = history['Student randomised box test accuracy']
        train = history['Student train accuracy']
        test = history['Student test accuracy']
        B_test = history['Student box test accuracy']
        P_test = history['Student plain test accuracy']
        LR = history['LR']
        error = history['Student-teacher error']
        run_histories.append((step_values, train, test, B_test, P_test, BR_test, LR, error))

if __name__ == "__main__":
    wandb_plot_CIFAR100('Student (debug)', 1, 1, 0)