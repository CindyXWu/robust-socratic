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
import datetime

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)
# Set the style and color palette
sns.set_style("whitegrid")
sns.set_palette("pastel")
plt.rcParams['figure.facecolor'] = '#C7D9FF'


def wandb_get_data(project_name: str, 
                   t_num: int, 
                   s_num: int, 
                   exp_dict: dict[str, List],
                   groupby_metrics: List[str],
                   s_mech: Optional[int] = None,
                   t_mech: Optional[int] = None,
                   loss_num: Optional[int] = None,
                   use_t_mech: Optional[bool] = False) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return historical information with mean/var interleaved."""
    runs = api.runs(project_name) 
    teacher = teacher_dict[t_num]
    student = student_dict[s_num]
    t_mech = list(exp_dict.keys())[t_mech].split(":")[-1].strip() if t_mech is not None else None
    s_mech = list(exp_dict.keys())[s_mech].split(":")[-1].strip() if s_mech is not None else None
    loss = loss_dict[loss_num]
    filtered_runs = []

    # Filter by above settings and remove any crashed or incomplete runs
    cutoff_date = datetime.datetime(2023, 5, 17)
    for run in runs:
        if "_timestamp" not in run.summary.keys():
            continue
        if (run.config.get('loss') == loss and
            run.config.get('teacher_mechanism') == t_mech and
            # run.state != 'running' and
            datetime.datetime.fromtimestamp(run.summary["_timestamp"]) >= cutoff_date
            ):
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= 10:
                filtered_runs.append(run)
                # Clean history of NaNs
                history = clean_history(history)
                # Filter history
                run.history = smooth_history(history)

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
        # **Use shortened names for mechs**
        if use_t_mech:
            combined['Group Name'] = [('T: '+new_mech_map[key[0]]+', S: '+new_mech_map[key[1]])] * len(combined)
        else:
            combined['Group Name'] = [('S: '+new_mech_map[key[1]])] * len(combined)
        histories.append(combined)
        print(combined['Group Name'])
    # histories = get_histories(grouped_runs)
    assert len(histories) is not None
    return histories


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


def get_order_list() -> Tuple[List, List]:
    """Order of subplots by metric name - used to make grouped plots make sense"""
    const_graph_list = ['T-S Top 1 Fidelity', 'T-S KL', 'T-S Test Difference']
    order_list = metric_names + const_graph_list
    return order_list


def custom_sort(col: str, type: str) -> int:
    """Used to sort the order of the subplots in the grouped plots."""
    order_list = get_counterfactual_order_list()
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
    n_cols = min(3, len(cols))
    n_rows = np.ceil(n_metrics / n_cols).astype(int)
    plot_width, plot_height = 20, 5* n_rows

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

    for i, mean_col in enumerate(cols):
        var_col = mean_col.replace(' Mean', ' Var')
        for line_num, history in enumerate(histories):
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
        if mode == 'acc' or mode == 'fidelity':
            axs[i].set_ylim(0, 100)
        else:
            axs[i].set_ylim(0, 7)
        axs[i].set_title(plot_names[i], fontsize=15)
        labelLines(axs[i].get_lines(), align=False, fontsize=11)
        axs[i].set_xlabel('Training step/100 iterations', fontsize=15)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=num_groups, fontsize=18, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontsize=20)
    plt.tight_layout(pad=2)
    plt.savefig('images/vstime/'+title+'.png', dpi=300, bbox_inches='tight')


def counterfactual_plot(histories: pd.DataFrame, exp_dict: Dict[str, List], title: str) -> None:
    """For a given run, plot counterfactual test accuracy, KL and top-1 fidelity on different plots."""
    acc_mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in exp_dict]
    kl_mean_cols = [col for col in histories[0].columns if col.replace(' T-S KL Mean', '') in exp_dict]
    top1_mean_cols = [col for col in histories[0].columns if col.replace(' T-S Top 1 Fidelity Mean', '') in exp_dict]
    acc_mean_cols.sort(key=lambda col: custom_sort(col, 'acc'))
    kl_mean_cols.sort(key=lambda col: custom_sort(col, 'kl'))
    top1_mean_cols.sort(key=lambda col: custom_sort(col, 'fidelity'))

    make_plot(histories, acc_mean_cols, "Counterfactual Test Accuracy "+title, mode='acc')
    make_plot(histories, kl_mean_cols, "Counterfactual T-S KL "+title, mode='kl')
    make_plot(histories, top1_mean_cols, "Counterfactual T-S Top 1 Fidelity "+title, mode='fidelity')


def plot_counterfactual_heatmaps(combined_history: List[pd.DataFrame], exp_dict: Dict[str, List], loss_num: int) -> Dict[str, np.ndarray]:
    data_to_plot = {}
    axes_labels = []
    num_teachers = 3
    num_students = 3
    loss = loss_dict[loss_num]

    for key in exp_dict.keys():
        data_to_plot[key] = np.zeros((num_students, num_teachers))
        axes_labels.append(key.replace('_', ' '))
                           
    for history in combined_history:
        print("cols", history.columns)
        name = history['Group Name'].iloc[0]
        mechs = name.split(', ')
        print(mechs)
        row = new_mech_list.index(mechs[1].replace('S: ', ''))
        col = new_mech_list.index(mechs[0].replace('T: ', ''))
        print(row, col)
        for key in exp_dict.keys():
            data_to_plot[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]

    for key, data in data_to_plot.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax)
        
        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=8)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=8)
        ax.set_xlabel('Teacher Training Mechanism')
        ax.set_ylabel('Student Distillation Training Mechanism')
        ax.set_title(f'Counterfactual {key.replace("_", " ")} Test Accuracy - {loss} Loss')
        plt.savefig(f'images/heatmaps/{loss}/'+key.replace(" ", "_")+'.png', dpi=300, bbox_inches='tight')
        plt.show()


def get_counterfactual_order_list():
    """Order of subplots by metric name - used to make grouped plots make sense"""
    const_graph_list = ['T-S Top 1 Fidelity', 'T-S KL', 'T-S Test Difference']
    order_list = counterfactual_metric_names + const_graph_list
    return order_list



plot_names = ['M S1 S2', 'S1 S2', 'Rand S1', 'Rand S2', 'Rand M']
teacher_mechs = ['M=100% S1=0% S2=0%', 'M=100% S1=0% S2=60%', 'M=100% S1=30% S2=60%']
if __name__ == "__main__":
    t_mech = 1
    use_t_mech = False
    loss_num = 0

    if not use_t_mech:
        title = f'{loss_dict[loss_num]} Loss, Teacher Mechanism ' + teacher_mechs[t_mech]
    else:
        title = f'{loss_dict[loss_num]} Loss '

    metric_names = [x.split(":")[-1].strip() for x in list(exp_dict_all.keys())]
    counterfactual_metric_names = [x.split(":")[-1].strip() for x in list(counterfactual_dict_all.keys())]
    loss = loss_dict[loss_num]

    # IMPORTANT: teacher mechanism must go first in the groupby_metrics list
    histories = wandb_get_data('Distill ResNet18_AP ResNet18_AP_Dominoes', t_num=1, s_num=1, exp_dict=exp_dict_all, groupby_metrics=['teacher_mechanism','student_mechanism'], t_mech=t_mech, loss_num=loss_num, use_t_mech=use_t_mech)
    counterfactual_plot(histories, counterfactual_dict_all, title)
    # plot_counterfactual_heatmaps(histories, exp_dict=exp_dict_all, loss_num=loss_num)