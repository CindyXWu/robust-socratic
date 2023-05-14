from collections import defaultdict
import pandas as pd
import numpy as np
import os
import einops
import torch
import multiprocessing as mp
from functools import reduce
import warnings
from typing import List, Optional, Dict, Tuple

from info_dicts import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from torchvision.utils import make_grid
import wandb
from wandb.sdk.wandb_run import Run
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore", category=FutureWarning)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)


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


def visualise_features_3d(s_features: torch.Tensor, t_features: torch.Tensor, title: Optional[str] = None):
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
                   groupby_metrics: List[str],
                   s_exp_num: Optional[int] = None,
                   t_exp_num: Optional[int] = None,
                   loss_num: Optional[int] = None) -> pd.DataFrame:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return historical information with mean/var interleaved."""
    runs = api.runs(project_name) 
    teacher = teacher_dict[t_num]
    student = student_dict[s_num]
    t_mech = list(exp_dict.keys())[t_exp_num] if t_exp_num else None
    s_mech = list(exp_dict.keys())[s_exp_num] if s_exp_num else None
    loss = "Jacobian" #loss_dict[loss_num]
    filtered_runs = []

    # Filter by above settings and remove any crashed or incomplete runs
    for run in runs:
        if (run.config.get('teacher') == teacher and 
            run.config.get('student') == student and
            run.config.get('loss') == loss):
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 100:
                filtered_runs.append(run)
                # Clean history of NaNs
                run.history = clean_history(history)

    assert(len(filtered_runs) > 0), "No runs found with the given settings"
    grouped_runs = get_grouped_runs(filtered_runs, groupby_metrics)

    # Compute the means and variances for all of the metrics for each group of runs
    histories = []
    for key, runs in grouped_runs.items():
        metrics = defaultdict(list)
        for run in runs:
            history = run.history
            for metric in history.columns:
                metrics[metric].append(history[[metric]])

        # Calculate the mean and variance for each metric between repeat runs
        means_and_vars = {}
        for metric, metric_values in metrics.items():
            combined = pd.concat(metric_values)
            mean = combined.groupby(combined.index)[metric].mean().reset_index().rename(columns={metric: f'{metric} Mean'})
            var = combined.groupby(combined.index)[metric].var().reset_index().rename(columns={metric: f'{metric} Var'})
            means_and_vars[metric] = mean.merge(var, left_index=True, right_index=True)

        # Combine the means and vars for each metric into a single dataframe
        first_metric = list(means_and_vars.keys())[0]
        combined = means_and_vars[first_metric]
        for metric in list(means_and_vars.keys())[1:]:
            combined = combined.merge(means_and_vars[metric], left_index=True, right_index=True)

        # Name each row of the dataframe with the values of the grouped metrics
        combined['name'] = [' '.join([str(k) for k in key])] * len(combined)
        histories.append(combined)

    # histories = get_histories(grouped_runs)
    assert(len(histories) > 0), "Something went wrong with making history dataframe"
    return histories


def plot_counterfactual_heatmaps(combined_history: List[pd.DataFrame], exp_dict: Dict[str, List]) -> Dict[str, np.ndarray]:
    data_to_plot = {}
    axes_labels = []
    num_keys = len(exp_dict.keys())

    for key in exp_dict.keys():
        data_to_plot[key] = np.zeros((num_keys, num_keys))
        axes_labels.append(key.replace('_', ' '))
                           
    for history in combined_history:
        name = history['name'].iloc[0]
        # Split name
        mechs= name.split(' ')
        row = list(exp_dict.keys()).index(mechs[1])
        col = list(exp_dict.keys()).index(mechs[0])
        for key in exp_dict.keys():
            data_to_plot[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]

    for key, data in data_to_plot.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax)
        
        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=8)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=8)
        ax.set_xlabel('Teacher Mechanism')
        ax.set_ylabel('Student Mechanism')
        ax.set_title(key)
        plt.show()


def compute_mean_and_variance(run: Run, metric: str):
    """Compute the mean and variance for a specific metric in a run."""
    metric_values = run.history[[metric]]
    mean = metric_values.groupby(metric_values.index)[metric].mean()
    mean = mean.reset_index().rename(columns={metric: f'{metric} Mean'})
    var = metric_values.groupby(metric_values.index)[metric].var()
    var = var.reset_index().rename(columns={metric: f'{metric} Var'})
    return mean.merge(var, on=metric_values.index)


def process_runs(runs: List[Run]):
    """Compute the mean and variance for all metrics in a group of runs."""
    metrics = runs[0].history.columns
    means_and_vars = {metric: compute_mean_and_variance(run, metric) for run in runs for metric in metrics}
    return reduce(lambda df1, df2: df1.merge(df2, on=df1.index), means_and_vars.values())


def get_histories(grouped_runs: Dict[Tuple[str, ...], List[Run]]) -> List[pd.DataFrame]:
    """Compute the mean and variance for all metrics for each group of runs."""
    histories = []
    for key, runs in grouped_runs.items():
        combined = process_runs(runs)
        combined['name'] = [' '.join([str(k) for k in key])] * len(combined)
        histories.append(combined)
    return histories


def get_grouped_runs(runs: List[Run], groupby_metrics: List[str]) -> Dict[Tuple[str, ...], List[Run]]:
    """Given runs list, set key to value of metrics used to group by and values to list of runs satisfyinh these metric values."""
    grouped_runs = defaultdict(list)
    for run in runs:
        key = tuple([run.config.get(m) for m in groupby_metrics])
        grouped_runs[key].append(run)
    return grouped_runs


def clean_history(history: pd.DataFrame) -> pd.DataFrame:
    """Remove any NaN datapoints individually due to data logging bug where wandb believes logging data as separate dictionaries is a new timstep at each call."""
    history_clean = pd.DataFrame()
    for col in history.columns:
        if not col.startswith('_'):
            col_array = history[col].values
            col_array_clean = col_array[~np.isnan(col_array)]
            history_clean[col] = pd.Series(col_array_clean)
    history_clean.reset_index(drop=True, inplace=True)
    return history_clean


def get_order_list(exp_dict: Dict[str, List]):
    """Order of subplots by metric name - used to make grouped plots make sense"""
    const_graph_list = ['T-S Top 1 Fidelity', 'T-S KL', 'T-S Test Difference', 'S Test', 'S Train', 'S Loss']
    dataset_specific_mechs = list(exp_dict.keys())
    order_list = dataset_specific_mechs + const_graph_list
    return order_list


def wandb_plot(histories: pd.DataFrame, title: str):
    sns.set(style='whitegrid', context='paper', font_scale=1.2)     # Set seaborn styling
    num_groups = len(set([history['name'].iloc[0] for history in histories]))
    palette = sns.color_palette("deep", num_groups)

    mean_cols = [col for col in histories[0].columns if col.endswith(' Mean')] 
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
        var_col = mean_col.replace(' Mean', ' Var')
        for history in histories:
            group_name = history['name'].iloc[0]
            if mean_col in history.columns and var_col in history.columns:
                axs[i].plot(history.index, history[mean_col], linewidth=1, label=group_name)
                axs[i].fill_between(history.index, 
                                    history[mean_col] - history[var_col].apply(np.sqrt),
                                    history[mean_col] + history[var_col].apply(np.sqrt),
                                    alpha=0.2)
        axs[i].set_title(mean_col.replace(' Mean', ''), fontsize=12)

    axs[-1].set_xlabel('Training step/100 iterations', fontsize=12)
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=4)

    fig.suptitle(title, fontsize=15)
    # plt.savefig(name+'.png', dpi=300, bbox_inches='tight')
    plt.show()


def custom_sort(col):
    metric_name = col.replace(' Mean', '')
    if metric_name in order_list:
        return order_list.index(metric_name)
    else:
        return len(order_list) + 1

if __name__ == "__main__":
    title = 'Jacobian loss'
    histories = wandb_get_data('Distill ResNet18_AP ResNet18_AP_Dominoes', t_num=1, s_num=1, exp_dict=dominoes_exp_dict, groupby_metrics=['teacher_mechanism','student_mechanism'], loss_num=1)
    order_list = get_order_list(dominoes_exp_dict)
    plot_counterfactual_heatmaps(histories, dominoes_exp_dict)