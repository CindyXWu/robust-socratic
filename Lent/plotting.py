import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
from collections import defaultdict
import pandas as pd
import numpy as np
from info_dicts import *
import seaborn as sns

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

def wandb_get_data(project_name, t_num, s_num, exp_num):
    runs = api.runs(project_name)
    teacher = 'ResNet18_CIFAR100' #teacher_dict[t_num]
    student = 'ResNet18_Flexi' #student_dict[s_num]
    # grouped_runs = defaultdict(list)
    t_mech = 'plain' #exp_dict[exp_num]
    s_mech = 'plain' #exp_dict[exp_num]
    # Filter
    runs = [run for run in runs if 
            run.config.get('teacher') == teacher and 
            run.config.get('student') == student and 
            run.config.get('teacher_mechanism') == t_mech
            ]
    histories = []
    i = 0
    for run in runs:
        history = run.history()
        histories.append(history)
        i += 1
        if i == 3:
            break
    return histories

def wandb_get_data(project_name, t_num, s_num, exp_num, groupby_metrics):
    runs = api.runs(project_name)
    teacher = 'ResNet18_CIFAR100' #teacher_dict[t_num]
    student = 'ResNet18_Flexi' #student_dict[s_num]
    t_mech = 'plain' #exp_dict[exp_num]
    s_mech = 'plain' #exp_dict[exp_num]

    # Filter
    filtered_runs = []
    for run in runs:
        if (run.config.get('teacher') == teacher and 
            run.config.get('student') == student and 
            run.config.get('teacher_mechanism') == t_mech):
            
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 20:
                filtered_runs.append(run)
    runs = filtered_runs

    # Group the runs by the values of specific metrics
    grouped_runs = defaultdict(list)
    for run in runs:
        key = tuple([run.summary.get(m) for m in groupby_metrics])
        grouped_runs[key].append(run)

    # Compute the means and variances for all of the metrics for each group of runs
    histories = []
    for key, runs in grouped_runs.items():
        metrics = defaultdict(list)
        for run in runs:
            history = run.history()
            for metric in history.columns:
                if not metric.startswith('_'):
                    metrics[metric].append(history[['_step', metric]])

        # Calculate the mean and variance for each metric between repeat runs
        means_and_vars = {}
        for metric, metric_values in metrics.items():
            combined = pd.concat(metric_values)
            mean = combined.groupby('_step')[metric].mean().reset_index().rename(columns={metric: f'{metric}_mean'})
            var = combined.groupby('_step')[metric].var().reset_index().rename(columns={metric: f'{metric}_var'})
            means_and_vars[metric] = mean.merge(var, on='_step')

        # Combine the means and vars for each metric into a single dataframe
        first_metric = list(means_and_vars.keys())[0]
        combined = means_and_vars[first_metric]
        for metric in list(means_and_vars.keys())[1:]:
            combined = combined.merge(means_and_vars[metric], on='_step')

        combined['name'] = [' '.join([str(k) for k in key])] * len(combined)
        histories.append(combined)

    return histories

# def wandb_plot(histories):
#     # Get a list of all the metrics across all the history dataframes
#     all_metrics = set()
#     for history in histories:
#         all_metrics.update([key for key in history.columns.tolist() if not key.startswith('_')])

#     # Determine the number of rows and columns needed for the subplots
#     n_metrics = len(all_metrics)
#     n_cols = min(2, n_metrics)
#     n_rows = np.ceil(n_metrics / n_cols).astype(int)

#     # Create a grid of subplots
#     fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8 * n_rows), sharex=True)

#     # Flatten the axs array so that we can iterate over it with a single loop
#     axs = axs.flatten()

#     # Remove any unused subplots
#     if n_metrics < n_rows * n_cols:
#         for i in range(n_metrics, n_rows * n_cols):
#             fig.delaxes(axs[i])

#     # Plot each metric on the corresponding subplot
#     for i, metric in enumerate(all_metrics):
#         for history in histories:
#             if metric in history.columns:
#                 axs[i].plot(history['_step'], history[metric])
#         axs[i].set_title(metric)
#         axs[i].legend()

#     # Set the x-axis label on the last subplot
#     axs[-1].set_xlabel('Step')

#     # Show the plot
#     plt.show()

def wandb_plot(histories, title):
    # Set seaborn styling
    sns.set(style='whitegrid', context='paper', font_scale=1.2)

    # Get a list of all the metrics across all the history dataframes
    all_metrics = set()
    for history in histories:
        all_metrics.update([key for key in history.columns.tolist() if not key.startswith('_')])

    # Determine the number of rows and columns needed for the subplots
    n_metrics = len(all_metrics)
    n_cols = min(2, n_metrics)
    n_rows = np.ceil(n_metrics / n_cols).astype(int)

    # Create a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8 * n_rows), sharex=True)

    # Flatten the axs array so that we can iterate over it with a single loop
    axs = axs.flatten()

    # Remove any unused subplots
    if n_metrics < n_rows * n_cols:
        for i in range(n_metrics, n_rows * n_cols):
            fig.delaxes(axs[i])

    # Plot each metric on the corresponding subplot
    for i, metric in enumerate(all_metrics):
        for history in histories:
            if metric in history.columns:
                axs[i].plot(history['_step'], history[metric], linewidth=2)
        axs[i].set_title(metric.capitalize(), fontweight='bold')
        axs[i].set_ylabel(metric)
        axs[i].legend()

    # Set the x-axis label on the last subplot
    axs[-1].set_xlabel('Step')

    # Adjust the layout for better spacing between subplots
    fig.tight_layout()

    # Set the title
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Save the plot as a high-resolution image
    plt.savefig('publication_quality_plot.png', dpi=300)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    title = 'ResNet18_CIFAR100 plain to ResNet18_Flexi'
    histories = wandb_get_data('Student (debug)', 1, 1, 0, ['spurious_corr'])
    wandb_plot(histories, title)