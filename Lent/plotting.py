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

def wandb_get_data(project_name, t_num, s_num, exp_num, groupby_metrics):
    runs = api.runs(project_name)
    teacher = 'ResNet18_CIFAR100' #teacher_dict[t_num]
    student = 'ResNet18_Flexi' #student_dict[s_num]
    t_mech = 'plain' #exp_dict[exp_num]
    s_mech = 'plain' #exp_dict[exp_num]
    loss = 'Base Distillation'
    filtered_runs = []
    # Filter by above settings and remove any crashed or incomplete runs
    for run in runs:
        if (run.config.get('teacher') == teacher and 
            run.config.get('student') == student and 
            run.config.get('teacher_mechanism') == t_mech and
            run.config.get('loss') == loss):
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 20:
                filtered_runs.append(run)
    runs = filtered_runs

    # Group the runs by the values of specific metrics
    grouped_runs = defaultdict(list)
    for run in runs:
        key = tuple([run.config.get(m) for m in groupby_metrics])
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

        # Name each row of the dataframe with the values of the grouped metrics
        combined['name'] = [' '.join([str(k) for k in key])] * len(combined)
        histories.append(combined)

    print(combined['name'])
    print(len(histories))
    return histories

def wandb_plot(histories, title):
    # Set seaborn styling
    sns.set(style='whitegrid', context='paper', font_scale=1.2)
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Helvetica'

    palette = sns.color_palette("tab10")

    # Get the columns that end in '_mean'
    mean_cols = [col for col in histories[0].columns if col.endswith('_mean')]

    # Determine the number of rows and columns needed for the subplots
    n_metrics = len(mean_cols)
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

    # Set a consistent color cycle for all subplots
    color_dict = {}
    color_cycle = iter(palette)
    for history in histories:
        group_name = history['name'].iloc[0]
        if group_name not in color_dict:
            color_dict[group_name] = next(color_cycle)
    for ax in axs:
        ax.set_prop_cycle(color=[color_dict[group_name] for group_name in color_dict])

    for i, mean_col in enumerate(mean_cols):
        var_col = mean_col.replace('_mean', '_var')
        for history in histories:
            group_name = history['name'].iloc[0]
            if mean_col in history.columns and var_col in history.columns:
                axs[i].plot(history['_step'], history[mean_col], linewidth=0.5, label=group_name)
                axs[i].fill_between(history['_step'], 
                                    history[mean_col] - 2 * history[var_col].apply(np.sqrt),
                                    history[mean_col] + 2 * history[var_col].apply(np.sqrt),
                                    alpha=0.2)
        axs[i].set_title(mean_col.replace('_mean', '').capitalize(), fontsize=12)
    
    # Add a legend to the right of each row
    for i in range(n_rows):
        handles, labels = axs[i*n_cols].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', fontsize=10)

    # Global x axis label
    axs[-1].set_xlabel('Step', fontsize=12)
    fig.suptitle(title, fontsize=15)
    # plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    title = 'ResNet18_CIFAR100 plain to ResNet18_Flexi'
    histories = wandb_get_data('Student (debug)', 1, 1, 0, ['spurious_corr', 'student_mechanism'])
    wandb_plot(histories, title)