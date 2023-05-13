from collections import defaultdict
import pandas as pd
import numpy as np
import os
import einops
import torch
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from torchvision.utils import make_grid
import wandb
from sklearn.manifold import TSNE


from info_dicts import *
os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

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
    

def show_images_grid(imgs_, class_labels, num_images, title=None):
    """Now modified to show both [c h w] and [h w c] images."""
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            img = imgs_[ax_i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            ax.imshow(img, cmap='Greys_r', interpolation='nearest')
            ax.set_title(f'Class: {class_labels[ax_i]}')  # Display the class label as title
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    if title:
        plt.savefig(image_dir+title+'.png')
    else:
        plt.show()


def visualise_features_3d(s_features, t_features, title=None):
    tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    s = tsne.fit_transform(s_features.detach().numpy())
    t = tsne.fit_transform(t_features.detach().numpy())
    s_x, s_y, s_z = s[:, 0], s[:, 1], s[:, 2]
    t_x, t_y, t_z = t[:, 0], t[:, 1], t[:, 2]

    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s_x, s_y, s_z, c='blue', label='Student', alpha=0.5)
    ax.scatter(t_x, t_y, t_z, c='red', label='Teacher', alpha=0.5)
    if title:
        plt.savefig(image_dir+"3d/"+title+'.png')
    else:
        plt.show()


def visualise_features_2d(s_features: torch.Tensor, t_features: torch.Tensor, title=None):
    """Student and teacher features shape [num_samples, feature_dim]."""
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
    s = tsne.fit_transform(s_features.detach().numpy())
    t = tsne.fit_transform(t_features.detach().numpy())
    s_x, s_y = s[:, 0], s[:, 1]
    t_x, t_y = t[:, 0], t[:, 1]

    fig = plt.figure(figsize=(5, 5), dpi=300)
    plt.scatter(s_x, s_y, c='blue', label='Student', alpha=0.5)
    plt.scatter(t_x, t_y, c='red', label='Teacher', alpha=0.5)
    if title:
        plt.savefig(image_dir+"2d/"+title+'_2d.png')
    else:
        plt.show()


def plot_images(dataloader, num_images, title=None):
    for i, (x, y) in enumerate(dataloader):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        show_images_grid(x, y, num_images, title=title)
        break



def wandb_get_data(project_name: str, 
                   t_num: int, 
                   s_num: int, 
                   exp_dict: dict[str, List],
                   s_exp_num: Optional[int] = None,
                   t_exp_num: Optional[int] = None,
                   loss_num: Optional[int] = None, 
                   groupby_metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return historical information with mean/var interleaved."""
    runs = api.runs(project_name) 
    teacher = teacher_dict[t_num]
    student = student_dict[s_num]
    t_mech = list(exp_dict.keys())[t_exp_num]
    s_mech = list(exp_dict.keys())[s_exp_num]
    loss = loss_dict[loss_num]
    filtered_runs = []

    # Filter by above settings and remove any crashed or incomplete runs
    for run in runs:
        if (run.config.get('teacher') == teacher and 
            run.config.get('student') == student and 
            run.config.get('loss') == loss):
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 100:
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

    return histories

# Order of subplots by metric name - used to make grouped plots make sense
order_list = ['Student train accuracy', 'Student test accuracy', 'Student plain test accuracy', 'Student randomised box test accuracy', 'Student box test accuracy', 'Student-teacher error', 'Student lr', 'Student loss']
def wandb_plot(histories: pd.DataFrame, title: str):
    sns.set(style='whitegrid', context='paper', font_scale=1.2)     # Set seaborn styling
    num_groups = len(set([history['name'].iloc[0] for history in histories]))
    palette = sns.color_palette("deep", num_groups)

    # Get the columns that end in '_mean'
    mean_cols = [col for col in histories[0].columns if col.endswith('_mean')] 
    mean_cols.sort(key=custom_sort)
    # Determine the number of rows and columns needed for the subplots
    n_metrics = len(mean_cols)
    n_cols = min(2, len(mean_cols))
    n_rows = np.ceil(n_metrics / n_cols).astype(int)

    # Create a grid of subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 8 * n_rows), sharex=True)
    axs = axs.flatten() # Flatten the array of subplots    # Flatten the axs array so that we can iterate over it with a single loop
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
                axs[i].plot(history['_step'], history[mean_col], linewidth=1, label=group_name)
                axs[i].fill_between(history['_step'], 
                                    history[mean_col] - 2 * history[var_col].apply(np.sqrt),
                                    history[mean_col] + 2 * history[var_col].apply(np.sqrt),
                                    alpha=0.2)
        axs[i].set_title(mean_col.replace('_mean', '').capitalize(), fontsize=12)

    axs[-1].set_xlabel('Step', fontsize=12)
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=4)

    fig.suptitle(title, fontsize=15)
    # plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def custom_sort(col):
    metric_name = col.replace('_mean', '')
    if metric_name in order_list:
        return order_list.index(metric_name)
    else:
        return len(order_list) + 1

if __name__ == "__main__":
    title = 'Jacobian loss'
    histories = wandb_get_data('Student (debug)', t_num=1, s_num=1, exp_dict=dominoes_exp_dict, loss_num=1)
    wandb_plot(histories, title)