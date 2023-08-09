from collections import defaultdict
import pandas as pd
import numpy as np
import os
from types import SimpleNamespace
import yaml
import logging
import wandb
from wandb.sdk.wandb_run import Run

import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines
from scipy.signal import savgol_filter
import seaborn as sns
from typing import List, Optional, Dict, Tuple

from config_setup import ConfigGroups, DistillConfig


os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)

sns.set_style("whitegrid")
sns.set_palette("pastel")


def heatmap_get_data(project_name: str,
                     loss_name: str,
                     groupby_metrics: List[str],) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    
    Calculate mean/var of metrics and return a list of history dataframes with mean/var interleaved.
    """
    runs = api.runs(project_name) 
    filtered_runs = []
    min_step = 300 # Filter partially logged/unfinished runs

    # Don't filter for teacher experiment type for heatmap data
    # Filter for loss and remove crashed/incomplete runs
    for run in runs:
        if run.config.get('distill_loss_type') == loss_name:
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= min_step:
                filtered_runs.append(run)
                history = drop_image_columns(history) # Remove artifacts
                history = clean_history(history) # Remove NaNs
                #history = smooth_history(history)
                run.history  = history
    # Check list of runs is not empty
    assert(len(filtered_runs) > 0), "No runs found with the given settings"
    
    """Key of form: tuple of groupby metrics (in order it's passed in, in groupby_metrics)"""
    grouped_runs: Dict = get_grouped_runs(filtered_runs, groupby_metrics)
    assert all(key is not None for key in grouped_runs.keys()), "Key is None"
    
    # Compute means/var for all metrics for each group of runs
    histories = create_histories_list(grouped_runs, mode='exhaustive')
    return histories


def create_histories_list(
    grouped_runs: Dict[tuple, List],
    mode: str,
    **kwargs) -> List[pd.DataFrame]:

    histories = []
    for key, runs in grouped_runs.items():
        metrics = defaultdict(list)
        for run in runs:
            history = run.history
            for metric in history.columns:
                metrics[metric].append(history[[metric]])

        means_and_vars_list = []
        for metric, metric_values in metrics.items():
            combined = pd.concat(metric_values)
            mean = combined.groupby(combined.index)[metric].mean().rename(f'{metric} Mean')
            var = combined.groupby(combined.index)[metric].var().rename(f'{metric} Var')
            means_and_vars_list.append(pd.concat([mean, var], axis=1))

        combined = pd.concat(means_and_vars_list, axis=1)

        if mode == 'exhaustive': # For heatmaps
            combined['Group Name'] = [{'T': key[0], 'S': key[1]}]*len(combined)
        elif mode == 'vstime': # For vstime - must pass in extra info via kwargs
            plot_tmechs_together = kwargs.get('plot_tmechs_together')
            if plot_tmechs_together is None:
                raise ValueError("plot_tmechs_together must be provided")
            if plot_tmechs_together:
                combined['Group Name'] = [{'T': key[0], 'S': key[1]}] * len(combined)
            else: # Student only in Group Name
                combined['Group Name'] = [key[1]] * len(combined)
        else: raise ValueError("Mode must be 'exhaustive' or 'vstime'")
                
        histories.append(combined)

    return histories
    
    
def wandb_get_data(project_name: str,
                   config: DistillConfig,
                   groupby_metrics: List[str],
                   plot_tmechs_together: Optional[bool] = False) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    For plotting over training time.
    
    Args:
        config: DistillConfig object containing experiment settings.
        groupby_metrics: List of metrics to group runs by (e.g. ["teacher mechanism", "student mechanism"]).
        plot_tmechs_together: Whether to plot teacher mechanisms together or separately. If plotted together, then each run also needs to be labelled with its teacher experiment type.
    """
    runs = api.runs(project_name) 
    filtered_runs = []
    min_step = 300 # Filter partially logged/unfinished runs

    for run in runs:
        if (run.config.get('model_type') == config.model_type and 
            run.config.get('experiment.name') == config.experiment.name and
            run.config.get('distill_loss_type') == config.distill_loss_type):
            try:
                history = run.history()
            except:
                history = run.history
            if '_step' in history.columns and history['_step'].max() >= min_step:
                filtered_runs.append(run)
                history = drop_image_columns(history) # Remove artifacts
                history = clean_history(history) # Remove NaNs
                run.history = smooth_history(history) # Smooth bumpy plots

    assert(len(filtered_runs) > 0), f"No runs found: teacher experiment {config.experiment.name} loss {config.distill_loss_type}"
    
    # Group filtered runs: key is tuple of values of metrics specified in groupby_metrics (e.g. "teacher mechanism"). Values = list of runs satisfying these metric values.
    grouped_runs: Dict = get_grouped_runs(filtered_runs, groupby_metrics)

    histories = create_histories_list(grouped_runs, mode='vstime', plot_tmechs_together=plot_tmechs_together)
    
    return histories


def plot_counterfactual_heatmaps(
    combined_history: List[pd.DataFrame], 
    exp_names: str,
    loss_name: str) -> Dict[str, np.ndarray]:
    """
    Args:
        exp_names: ordered list of experiment names to plot.
    """
    data_to_plot = {}
    axes_labels = []
    num_keys = len(exp_names)

    for i, key in enumerate(exp_names):
        data_to_plot[key] = np.zeros((num_keys, num_keys))
        axes_labels.append(exp_names[i])
                           
    for history in combined_history:
        mechs = history['Group Name'].iloc[0]
        row = exp_names.index(mechs['S'])
        col = exp_names.index(mechs['T'])
        for key in exp_names:
            data_to_plot[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]

    for key, data in data_to_plot.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax, vmax=100, vmin=0)
        
        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=15)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=15)
        ax.set_xlabel('Teacher Training Mechanism', fontsize=15)
        ax.set_ylabel('Student Training Mechanism', fontsize=15)
        # ax.set_title(f'Counterfactual {key.replace("_", " ")} Test Accuracy - {loss} Loss')
        plt.savefig(f'images/heatmaps/{loss_name}/{key}.png', dpi=300, bbox_inches='tight')


def drop_image_columns(df):
    """Where images have been logged, run.history will return artifacts also. 
    This function removes these artifacts from the history dictionary.
    """
    cols_to_drop = []
    for col in df.columns:
        if any(isinstance(i, dict) for i in df[col]):
            cols_to_drop.append(col)
    df = df.drop(columns=cols_to_drop)
    return df
            

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


def smooth_history(history: pd.DataFrame, 
                   window_length: Optional[int]=5, 
                   polyorder: Optional[int] = 3) -> pd.DataFrame:
    """Smooth each column in the DataFrame, interpolating for NaNs."""
    nan_mask = history.isna()
    filtered_history = history.interpolate().apply(lambda x: savgol_filter(x, window_length, polyorder, mode='mirror'))
    filtered_history[nan_mask] = np.nan # Apply the NaN mask to the filtered history
    
    return filtered_history

        
def get_grouped_runs(runs: List[Run], groupby_metrics: List[str]) -> Dict[Tuple[str, ...], List[Run]]:
    """Key = value of metrics specified in groupby_metrics (e.g. "teacher mechanism"). Values = list of runs satisfying these metric values."""
    grouped_runs = defaultdict(list)
    
    for run in runs:
        key = tuple([get_nested_value(run.config, m) for m in groupby_metrics])
        grouped_runs[key].append(run)
        
    return grouped_runs


def get_nested_value(d: dict, keys_str: str):
    """Helper function for the above get_grouped_runs function which deals with nested dictionaries."""
    keys = keys_str.split('.')
    for key in keys:
        if d is None or not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def get_order_list(exp_names: List) -> List:
    """Order of subplots by metric name - used to make grouped plots make sense"""
    const_graph_list = ['T-S Top 1 Fidelity', 'T-S KL', 'T-S Test Difference']
    order_list = exp_names + const_graph_list
    return order_list


def custom_sort(col: str, type: str) -> int:
    order_list = get_order_list(exp_names)
    
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


def make_plot(histories: List[pd.DataFrame], 
              cols: List[str], 
              title: str, 
              mode: str) -> None:
    """
    For plots over time.
    
    Args:
        histories: List of dataframes containing historical information for each group of runs.
        cols: List of column names to plot.
        mode: Plotting accuracy, KL or top-1.
    """
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
                # The key to matching the legend to lines is to check order of dominoes_exp_dict.keys() - these don't correspond to plotting order
                print("Group name: ", group_name, "Line num: ", line_num, "actual_group_names: ", actual_group_names[line_num])
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
    """Extract unique column group names and plot them on separate plots - for plotting over training time."""
    mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in exp_names]
    mean_cols.sort(key=lambda col: custom_sort(col, 'acc'))
    make_plot(histories, mean_cols, title, 'acc')


def recursive_namespace(data):
    """Unpack YAML file into dot notation indexable form."""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: recursive_namespace(v) for k, v in data.items()})
    return data


if __name__ == "__main__":
    # Somewhat immutable things
    exp_names = [config.name for config in ConfigGroups.exhaustive]
    actual_group_names = ["I", "A", "B", "AB", "IB", "IA", "IAB"] # Actual order of names here
    loss_names = ['BASE', 'JACOBIAN']
    
    # Configs - amazing part of using config YAML is I can load all settings in
    config_filename = "distill_config"
    with open(f"configs/{config_filename}.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    config = recursive_namespace(config)
    wandb_project_name = f"DISTILL {config.model_type} {config.dataset_type} {config.config_type}"
    
    # To be changed
    mode = 0 # 0 for heatmap, 1 for plots
    groupby_metrics = ["experiment.name","experiment_s.name"]

    if mode == 0:
        for loss_name in loss_names:
            # IMPORTANT: teacher mechanism must go first in the groupby_metrics list
            histories: List[pd.DataFrame] = heatmap_get_data(project_name=wandb_project_name, loss_name=loss_name, groupby_metrics=groupby_metrics)
        
            plot_counterfactual_heatmaps(histories, exp_names, loss_name)
            
    elif mode == 1:
        for t_exp_name in exp_names:
            for loss_name in loss_names:
                title = f"{config.model_type} {config.experiment.name} {config.distill_loss_type}"
                # IMPORTANT: teacher mechanism must go first in the groupby_metrics list
                histories = wandb_get_data(project_name=wandb_project_name, config=config, groupby_metrics=groupby_metrics, plot_tmechs_together=False)
                
                wandb_plot(histories, title)