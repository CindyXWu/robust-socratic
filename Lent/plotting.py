from collections import defaultdict
import pandas as pd
import numpy as np
import os
import einops
import torch
from functools import reduce
import warnings
from torch.utils.data import DataLoader
from labellines import labelLines
from typing import List, Optional, Dict, Tuple

from info_dicts import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
import seaborn as sns
from torchvision.utils import make_grid
import wandb
from wandb.sdk.wandb_run import Run
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)
# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette("pastel")
    

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


def plot_images(dataloader: DataLoader, num_images: int, title: Optional[str] = None):
    for i, (x, y) in enumerate(dataloader):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        show_images_grid(x, y, num_images, title=title)
        break


def heatmap_get_data(project_name: str, 
                   t_num: int, 
                   s_num: int, 
                   exp_dict: dict[str, List],
                   groupby_metrics: List[str],
                   s_mech: Optional[int] = None,
                   t_mech: Optional[int] = None,
                   loss_num: Optional[int] = None) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return historical information with mean/var interleaved."""
    runs = api.runs(project_name) 
    teacher = teacher_dict[t_num]
    student = student_dict[s_num]
    t_mech = list(exp_dict.keys())[t_mech] if t_mech != None else None
    s_mech = list(exp_dict.keys())[s_mech] if s_mech != None else None
    loss = loss_dict[loss_num]
    filtered_runs = []

    # Don't filter for t_mech when running for heatmap data
    # Filter by above settings and remove any crashed or incomplete runs
    for run in runs:
        if run.config.get('loss') == loss:
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 50:
                filtered_runs.append(run)
                # Clean history of NaNs
                history = clean_history(history)
                # Filter history
                #history = smooth_history(history)
                run.history  = history
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

        combined['Group Name'] = [('T: '+mech_map[key[0]]+', S: '+mech_map[key[1]])] * len(combined)
        histories.append(combined)

    # histories = get_histories(grouped_runs)
    return histories


def wandb_get_data(project_name: str, 
                   t_num: int, 
                   s_num: int, 
                   exp_dict: dict[str, List],
                   groupby_metrics: List[str],
                   s_mech: Optional[int] = None,
                   t_mech: Optional[int] = None,
                   loss_num: Optional[int] = None,
                   plot_tmechs_together: Optional[bool] = False) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return historical information with mean/var interleaved."""
    runs = api.runs(project_name) 
    teacher = teacher_dict[t_num]
    student = student_dict[s_num]
    t_mech = list(exp_dict.keys())[t_mech] if t_mech != None else None
    s_mech = list(exp_dict.keys())[s_mech] if s_mech != None else None
    loss = loss_dict[loss_num]
    filtered_runs = []

    for run in runs:
        if (run.config.get('teacher') == teacher and 
            run.config.get('student') == student and
            run.config.get('teacher_mechanism') == t_mech and
            run.config.get('loss') == loss):
            try:
                history = run.history()
            except:
                history = run.history
            if '_step' in history.columns and history['_step'].max() >= 10:
                filtered_runs.append(run)
                # Clean history of NaNs
                history = clean_history(history)
                # Filter history
                run.history = smooth_history(history)

    assert(len(filtered_runs) > 0), f'No runs found: tmech {t_mech} loss {loss}'
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
        # **Use shortened names for mechs**
        if plot_tmechs_together:
            combined['Group Name'] = [('T: '+mech_map[key[0]]+', S: '+mech_map[key[1]])] * len(combined)
        else:
            combined['Group Name'] = [('S: '+mech_map[key[1]])] * len(combined)
        histories.append(combined)

    # histories = get_histories(grouped_runs)
    return histories


def plot_counterfactual_heatmaps(combined_history: List[pd.DataFrame], exp_dict: Dict[str, List], loss_num: int) -> Dict[str, np.ndarray]:
    data_to_plot = {}
    axes_labels = []
    num_keys = len(exp_dict.keys())
    loss = loss_dict[loss_num]

    for i, key in enumerate(list(exp_dict.keys())):
        data_to_plot[key] = np.zeros((num_keys, num_keys))
        axes_labels.append(mech_map_names[i])
                           
    for history in combined_history:
        name = history['Group Name'].iloc[0]
        mechs= name.split(', ')
        row = mech_map_names.index(mechs[1].replace('S: ' , ''))
        col = mech_map_names.index(mechs[0].replace('T: ', ''))
        for key in exp_dict.keys():
            data_to_plot[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]

    for key, data in data_to_plot.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax, vmax=100, vmin=0)
        
        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=15)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=15)
        ax.set_xlabel('Teacher Training Mechanism', fontsize=15)
        ax.set_ylabel('Student Training Mechanism', fontsize=15)
        # ax.set_title(f'Counterfactual {key.replace("_", " ")} Test Accuracy - {loss} Loss')
        plt.savefig(f'images/heatmaps/{loss}/'+key+'.png', dpi=300, bbox_inches='tight')


def get_grouped_runs(runs: List[Run], groupby_metrics: List[str]) -> Dict[Tuple[str, ...], List[Run]]:
    """Key = value of metrics specified in groupby_metrics (e.g. "teacher mechanism"). Values = list of runs satisfying these metric values."""
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


def smooth_history(history: pd.DataFrame, window_length: Optional[int]=5, polyorder: Optional[int] = 3) -> pd.DataFrame:
    nan_mask = history.isna()
    # Smooth each column in the DataFrame, interpolating for NaNs
    filtered_history = history.interpolate().apply(lambda x: savgol_filter(x, window_length, polyorder, mode='mirror'))
    filtered_history[nan_mask] = np.nan # Apply the NaN mask to the filtered history
    return filtered_history


def get_order_list(exp_dict: Dict[str, List]) -> Tuple[List, List]:
    """Order of subplots by metric name - used to make grouped plots make sense"""
    const_graph_list = ['T-S Top 1 Fidelity', 'T-S KL', 'T-S Test Difference']
    dataset_specific_mechs = list(exp_dict.keys())
    order_list = dataset_specific_mechs + const_graph_list
    return order_list


def custom_sort(col: str, type: str) -> int:
    order_list = get_order_list(exp_dict)
    """Used to sort the order of the subplots in the grouped plots."""
    match type:
        case 'acc':
            metric_name = col.replace(' Mean', '')
        case 'kl':
            metric_name = col.replace(' T_S KL Mean', '')
        case 'fidelity':
            metric_name = col.replace(' T_S Top 1 Fidelity Mean', '')
    if metric_name in order_list:
        return order_list.index(metric_name)
    else:
        return len(order_list) + 1


def make_plot(histories: List[pd.DataFrame], cols: List[str], title: str, mode: str) -> None:
    sns.set(style='whitegrid', context='paper', font_scale=1)
    num_groups = len(set([history['Group Name'].iloc[0] for history in histories]))

    # Determine rows and columns for subplots
    n_metrics = len(cols)
    n_cols = min(2, len(cols))
    n_rows = np.ceil(n_metrics / n_cols).astype(int)
    plot_width, plot_height = 13, 3.75* n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(plot_width, plot_height), sharey=True)
    axs = axs.flatten() # Flatten the axs array so that we can iterate over it with a single loop
    # Remove any unused subplots
    if n_metrics < n_rows * n_cols:
        for i in range(n_metrics, n_rows * n_cols):
            fig.delaxes(axs[i])

    # Colour and line style stuff
    colors = mpl.cm.get_cmap('cool', num_groups+1) # +1 as we ditch yellow
    color_dict = {}
    for i, history in enumerate(histories):
        group_name = history['Group Name'].iloc[0]
        color = colors(i)
        if color[0] > 0.8 and color[1] > 0.8 and color[2] < 0.2:  # Check if i exceeds the maximum index of colors
            continue
        if group_name not in color_dict:
            color_dict[group_name] = colors(i)  # Use the next color in the colormap
    line_styles = ['-', '--', '-.', ':']

    for ax in axs:
        ax.set_prop_cycle(color=[color_dict[group_name] for group_name in color_dict])
    legend_handles = []
    for i, mean_col in enumerate(cols): # Iterate over subplots
        var_col = mean_col.replace(' Mean', ' Var')
        for line_num, history in enumerate(histories): # Iterate over lines
            group_name = history['Group Name'].iloc[0]
            if mean_col in history.columns and var_col in history.columns:
                line = axs[i].plot(history.index, 
                                   history[mean_col], 
                                   linewidth=3, 
                                   label=group_name, 
                                   color=color_dict[group_name], 
                                   linestyle=line_styles[line_num%len(line_styles)])
                axs[i].fill_between(history.index, 
                                    history[mean_col] - history[var_col].apply(np.sqrt),
                                    history[mean_col] + history[var_col].apply(np.sqrt),
                                    alpha=0.2)
                # the key to matching the legend to the lines is to check the order of dominoes_exp_dict.keys() cos I don't think these correspond to plotting order
                print("group_name: ", group_name, "line_num: ", line_num, "actual_group_names: ", actual_group_names[line_num])
                if i == 0:  # Only add legend handles once per group (correspond to first subplot)
                    legend_handles.append(mpl.lines.Line2D([0], [0], 
                                                           color=color_dict[group_name], 
                                                           linestyle=line_styles[line_num%len(line_styles)], 
                                                           label=actual_group_names[line_num],
                                                           linewidth=2))
        axs[i].set_title(mean_col.replace(' Mean', '').replace('_', ' '), fontsize=18)
        axs[i].tick_params(axis='both', which='major', labelsize=15)
        axs[i].tick_params(axis='both', which='minor', labelsize=12)
        axs[i].set_ylim(-10, 110)
        match mode:
            case 'acc':
                axs[i].set_ylabel('Test accuracy %', fontsize=15)
            case 'kl':
                axs[i].set_ylabel('KL divergence', fontsize=15)
            case 'fidelity':
                axs[i].set_ylabel('Top-1 fidelity %', fontsize=15)
        labelLines(axs[i].get_lines(), align=False, fontsize=13)
        axs[i].set_xlabel('Training step/100 iterations', fontsize=15)
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=legend_handles, loc='lower right', ncol=1, bbox_to_anchor=(0.8, 0.05), fontsize=18, frameon=False)
    # fig.suptitle(title, fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig('images/exhaustivevstime/'+title+'.png', dpi=300, bbox_inches='tight')


def counterfactual_plot(histories: pd.DataFrame, exp_dict: Dict[str, List], title: str) -> None:
    """For a given run, plot counterfactual test accuracy, KL and top-1 fidelity on different plots."""
    metric_names = list(exp_dict.keys())
    acc_mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in metric_names]
    kl_mean_cols = [col for col in histories[0].columns if col.replace(' T-S KL Mean', '') in metric_names]
    top1_mean_cols = [col for col in histories[0].columns if col.replace(' T-S Top 1 Fidelity Mean', '') in metric_names]
    acc_mean_cols.sort(key=lambda col: custom_sort(col, 'acc'))
    kl_mean_cols.sort(key=lambda col: custom_sort(col, 'kl'))
    top1_mean_cols.sort(key=lambda col: custom_sort(col, 'fidelity'))

    make_plot(histories, acc_mean_cols, "Counterfactual Test Accuracy"+title)
    make_plot(histories, kl_mean_cols, "Counterfactual T-S KL"+title)
    make_plot(histories, top1_mean_cols, "Counterfactual T-S Top 1 Fidelity"+title)


def wandb_plot(histories: List[pd.DataFrame], title: str) -> None:
    mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in list(dominoes_exp_dict.keys())] 
    mean_cols.sort(key=lambda col: custom_sort(col, 'acc'))
    make_plot(histories, mean_cols, title, 'acc')

actual_group_names = ['CIFAR10 MNIST Box', 'CIFAR10 Box', 'CIFAR10 MNIST', 'MNIST Box', 'Box', 'MNIST', 'CIFAR10']
if __name__ == "__main__":
    title = 'Jacobian Matching Distillation'
    mode = 1 # 0 for heatmap, 1 for plots
    loss_num = 1

    exp_dict = dominoes_exp_dict 
    loss_names = ['Base', 'Jacobian']

    if mode == 0:
        # IMPORTANT: teacher mechanism must go first in the groupby_metrics list
        histories = heatmap_get_data('Distill ResNet18_AP ResNet18_AP_Dominoes', t_num=1, s_num=1, exp_dict=dominoes_exp_dict, groupby_metrics=['teacher_mechanism','student_mechanism'], t_mech=2, loss_num=loss_num)
        plot_counterfactual_heatmaps(histories, dominoes_exp_dict, loss_num)
    elif mode == 1:
        for t_mech in range(7):
            for loss_num in range(2):
                title = f'{loss_names[loss_num]} {list(dominoes_exp_dict.keys())[t_mech]}'
                # IMPORTANT: teacher mechanism must go first in the groupby_metrics list
                histories = wandb_get_data('Distill ResNet18_AP ResNet18_AP_Dominoes', t_num=1, s_num=1, exp_dict=dominoes_exp_dict, groupby_metrics=['teacher_mechanism','student_mechanism'], t_mech=t_mech, loss_num=loss_num, plot_tmechs_together=False)
                wandb_plot(histories, title)