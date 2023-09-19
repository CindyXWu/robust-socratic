"""Common functions for quick visualisations."""
import torch
import torchvision
from torch.utils.data import DataLoader

import numpy as np
import os
import einops
from collections import defaultdict
import pandas as pd
import warnings
import math
import matplotlib.pyplot as plt
from wandb.sdk.wandb_run import Run
from scipy.signal import savgol_filter
from sklearn.manifold import TSNE
from typing import List, Optional, Dict, Tuple
from types import SimpleNamespace

warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)


def plot_PIL_batch(dataloader: DataLoader, num_images: int) -> None:
    """Returns PIL image of batch for later logging to WandB."""
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    grid = torchvision.utils.make_grid(images).numpy()
    grid = np.transpose(grid, (1, 2, 0))

    cols = round(math.sqrt(num_images))
    rows = math.ceil(num_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(16, 16))

    # Loop over all subplots and add images with labels
    for i, ax in enumerate(axs.flat):
        # Transpose image from (C, H, W) to (H, W, C) for plotting
        img = np.transpose(images[i].numpy(), (1, 2, 0))
        ax.imshow(img)
        ax.set_title("Class: " + str(labels[i].item()))
        ax.set_xticks([]), ax.set_yticks([])

    # Remove empty subplots
    if num_images < rows*cols:
        for idx in range(num_images, rows*cols):
            fig.delaxes(axs.flatten()[idx])
            
    return fig

    
def show_images_grid(imgs_, class_labels, num_images):
    """Now modified to show both [c h w] and [h w c] images."""
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            img = imgs_[ax_i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = einops.rearrange(img, 'c h w -> h w c')
            ax.imshow(img, cmap='Greys_r', interpolation='nearest')
            ax.set_title(f'Class: {class_labels[ax_i]}')  # Display the class label as title
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    print("showing plot")
    plt.show()


def plot_images(dataloader: DataLoader, num_images: int, title: Optional[str] = None):
    for i, (x, y) in enumerate(dataloader):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        show_images_grid(x, y, num_images, title=title)
        break


def visualise_features_3d(s_features: torch.Tensor, 
                          t_features: torch.Tensor, 
                          title: Optional[str] = None):
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
        plt.savefig(f'{image_dir}3d/{title}.png')
    else:
        plt.show()


def visualise_features_2d(s_features: torch.Tensor, 
                          t_features: torch.Tensor, 
                          title=None):
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
        plt.savefig(f'{image_dir}2d/{title}.png')
    else:
        plt.show()


## =========================================================================
# Helper functions for graph plotting from WandB
## =========================================================================

from collections import defaultdict
from typing import Dict, List
import pandas as pd

def create_histories_list(
    grouped_runs: Dict[tuple, List],
    mode: str,
    **kwargs) -> List[pd.DataFrame]:
    """Takes each metric and groups runs with the same ones, then calculates the mean and variance."""
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
            
            # Check the number of data points before attempting to compute variance
            if len(combined) > 1:
                var = combined.groupby(combined.index)[metric].var().fillna(0).rename(f'{metric} Var')
            else:
                var = pd.Series(0, index=combined.index, name=f'{metric} Var')
            
            means_and_vars_list.append(pd.concat([mean, var], axis=1))

        combined = pd.concat(means_and_vars_list, axis=1)

        if mode == 'exhaustive': # For heatmaps
            combined['Group Name'] = {'T': key[0], 'S': key[1]}
        elif mode == 'vstime': # For vstime - must pass in extra info via kwargs
            grid = kwargs.get('grid')
            if grid is None:
                raise ValueError("Whether to use grid must be provided")
            if grid:
                combined['Group Name'] = {'T': key[0], 'S': key[1]}
            else: # Student only in Group Name
                combined['Group Name'] = key[1]
        else: raise ValueError("Mode must be 'exhaustive' or 'vstime'")
                
        histories.append(combined)
        
    return histories


def drop_non_numeric_columns(df):
    """Drop columns that cannot be converted entirely to numeric values."""
    cols_to_drop = []
    
    for col in df.columns:
        try:
            pd.to_numeric(df[col])
        except:
            cols_to_drop.append(col)
                
    df = df.drop(columns=cols_to_drop)
    return df


def clean_history(history: pd.DataFrame) -> pd.DataFrame:
    """Remove any NaN datapoints individually due to data logging bug where wandb believes logging data as separate dictionaries is a new timstep at each call."""
    history_clean = pd.DataFrame()
    
    for col in history.columns:
        if not col.startswith('_'):
            col_array = history[col].values
            if np.issubdtype(col_array.dtype, np.number):  # Check if the data type is numeric
                col_array_clean = col_array[~np.isnan(col_array)]
                history_clean[col] = pd.Series(col_array_clean)
            else:
                # Handle non-numeric columns, e.g., copy them as is
                history_clean[col] = history[col]
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


def custom_sort(col: str, type: str, exp_names: List[str]) -> int:
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


def recursive_namespace(data):
    """Unpack YAML file into dot notation indexable form."""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: recursive_namespace(v) for k, v in data.items()})
    return data


def condition_for_similarity(s, t, key):
        return (s == t and s == key)


def condition_for_student(s, t, key):
    return all(k in s for k in key) and all(k not in t for k in key) and any(k in s and k in t for k in exp_names if k not in key)


def condition_for_teacher(s, t, key):
    return all(k in t for k in key) and all(k not in s for k in key) and any(k in s and k in t for k in exp_names if k not in key)


def condition_for_neither(s, t):
    return s != t


def save_df_csv(df: pd.DataFrame, title: str, head: int = None):
    """head: If value is valid integer then save head."""
    df.to_csv(f"run_data/{title}.csv", index=False) if head is None else df.head(head).to_csv(f"run_data/{title}.csv", index=False)