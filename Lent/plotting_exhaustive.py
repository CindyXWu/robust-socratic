import pandas as pd
import numpy as np
import os
import pickle
import wandb
import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines
import seaborn as sns
from typing import List, Dict

from config_setup import ConfigGroups, BoxPatternType, DistillLossType
from plotting_common import drop_non_numeric_columns, clean_history, smooth_history, get_grouped_runs, custom_sort, create_histories_list

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)


def heatmap_get_data(
    project_name: str,
    loss_name: str,
    box_pattern: str,
    groupby_metrics: List[str],
    additional_naming: str
) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return a list of history dataframes with mean/var interleaved.
    
    Args:
        config: DistillConfig object containing experiment settings.
        additional_naming: Additional string to add to the end of the file name.
        
    Returns:
        List[pd.DataFrame]: The dataframes for the teacher runs.
    """
    runs = api.runs(project_name)
        
    filtered_runs = []
    min_step = 1200 # Filter partially logged/unfinished runs

    # Filter for loss and correct experiment name, and remove crashed/incomplete runs
    for run in runs:
        if run.config.get('distill_loss_type') == loss_name and run.config.get("experiment", {}).get("name") in label_group_names:
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= min_step:
                history = drop_non_numeric_columns(history) # Remove artifacts
                history = clean_history(history) # Remove NaNs
                #history = smooth_history(history)
                run.history  = history
                filtered_runs.append(run)
              
    # Check list of runs is not empty
    assert(len(filtered_runs) > 0), "No runs found with the given settings"
    
    """Key of form: tuple of groupby metrics (in order it's passed in, in groupby_metrics)"""
    grouped_runs: Dict = get_grouped_runs(filtered_runs, groupby_metrics)
    
    # Compute means/var for all metrics for each group of runs
    histories: list[pd.DataFrame] = create_histories_list(grouped_runs, mode='exhaustive')
    
    file_name = f"run_data/heatmap {loss_name} {box_pattern} {additional_naming}"
    with open(file_name, "wb") as f:
        pickle.dump(histories, f)
        
    return histories

    
def wandb_get_data(
    project_name: str,
    t_exp_name: str,
    loss_name: str,
    model_name: str,
    box_pattern: str,
    groupby_metrics: List[str],
    grid: bool = True,
    min_step: int = 300
) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    For plotting over training time.
    
    Args:
        groupby_metrics: List of metrics to group runs by (e.g. ["teacher mechanism", "student mechanism"]).
        plot_tmechs_together: Whether to plot teacher mechanisms together or separately. If plotted together, then each run also needs to be labelled with its teacher experiment type.
        grid: If we want to plot all teachers on the same grid, then turn off filtering by teacher mechanism.
        min_step: Cutoff step for filtering unfinished runs.
    """
    runs = api.runs(project_name) 
    filtered_runs = []
    print(f"Running {t_exp_name} {loss_name}")
    
    for run in runs:
        # If grid is True, then ignore the run_exp filter.
        if grid or (not grid and run.config.get("experiment", {}).get("name") == t_exp_name):
            if run.config.get('model_type') == model_name and run.config.get('distill_loss_type') == loss_name:
                try: history = run.history()
                except: history = run.history
                if '_step' in history.columns and history['_step'].max() >= min_step:
                    history = drop_non_numeric_columns(history) # Remove artifacts
                    history = clean_history(history) # Remove NaNs
                    run.history = smooth_history(history) # Smooth bumpy plots
                    filtered_runs.append(run)
                
    assert(len(filtered_runs) > 0), f"No runs found: {t_exp_name} {loss_name}"
    
    # Group filtered runs: key is tuple of values of metrics specified in groupby_metrics (e.g. "teacher mechanism"). Values = list of runs satisfying these metric values.
    grouped_runs: Dict[tuple[str], list] = get_grouped_runs(filtered_runs, groupby_metrics)
    histories: list[pd.DataFrame] = create_histories_list(grouped_runs, mode='vstime', grid=grid)
    
    file_name = f"run_data/vstime {loss_name} {box_pattern} grid" if grid else f"vstime {t_exp_name} {loss_name}"
    with open(file_name, "wb") as f:
        pickle.dump(histories, f)
        
    return histories
    
    
def plot_counterfactual_heatmaps(
    combined_history: List[pd.DataFrame], 
    exp_names: str,
    loss_name: DistillLossType,
    box_pattern: BoxPatternType,
    type: str = 'acc',
    additional_naming: str = '') -> None:
    """
    Plot heatmap of final KL-divergence, top-1 fidelity or accuracy for each of 49 student-teacher mechanism combinations.
    
    Args:
        exp_names: ordered list of experiment names to plot.
        type: string, 'acc', 'kl' or 'top1'
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
            match type:
                case 'acc':
                    data_to_plot[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]
                case 'kl':
                    data_to_plot[key][row, col] = history[f'{key} T-S KL Mean'].loc[history[f'{key} T-S KL Mean'].last_valid_index()]
                case 'top1':
                    data_to_plot[key][row, col] = history[f'{key} T-S Top 1 Fidelity Mean'].loc[history[f'{key} T-S Top 1 Fidelity Mean'].last_valid_index()]

    for key, data in data_to_plot.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax, vmax=100, vmin=0)
        
        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=15)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=15)
        ax.set_xlabel('Teacher Training Mechanism', fontsize=15)
        ax.set_ylabel('Student Training Mechanism', fontsize=15)
        ax.set_title(f'{key.replace("_", " ")} {type} {loss_name}')
        plt.savefig(f"images/heatmaps/{loss_name}_{box_pattern}/{type}_{key}_{additional_naming}.png", dpi=300, bbox_inches='tight')


def plot_mean_variance_heatmaps(
    combined_history: List[pd.DataFrame], 
    exp_names: str,
    loss_name: DistillLossType,
    box_pattern: BoxPatternType,
    type: str = 'acc'
) -> None:
    """
    Plot heatmap of final KL-divergence, top-1 fidelity or accuracy for each of 49 student-teacher mechanism combinations, and their variances.
    
    Args:
        exp_names: ordered list of experiment names to plot.
        type: string, 'acc', 'kl' or 'top1'
    """
    print(f"plotting vars in heatmap loss {loss_name} box {box_pattern}")
    data_to_plot_mean = {}
    data_to_plot_variance = {}
    axes_labels = []
    num_keys = len(exp_names)

    for i, key in enumerate(exp_names):
        data_to_plot_mean[key] = np.zeros((num_keys, num_keys))
        data_to_plot_variance[key] = np.zeros((num_keys, num_keys))
        axes_labels.append(exp_names[i])
                           
    for history in combined_history:
        mechs = history['Group Name'].iloc[0]
        row = exp_names.index(mechs['S'])
        col = exp_names.index(mechs['T'])
        for key in exp_names:
            match type:
                case 'acc':
                    data_to_plot_mean[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]
                    data_to_plot_variance[key][row, col] = history[f'{key} Var'].loc[history[f'{key} Var'].last_valid_index()]
                case 'kl':
                    data_to_plot_mean[key][row, col] = history[f'{key} T-S KL Mean'].loc[history[f'{key} T-S KL Mean'].last_valid_index()]
                    data_to_plot_variance[key][row, col] = history[f'{key} T-S KL Var'].loc[history[f'{key} T-S KL Var'].last_valid_index()]
                case 'top1':
                    data_to_plot_mean[key][row, col] = history[f'{key} T-S Top 1 Fidelity Mean'].loc[history[f'{key} T-S Top 1 Fidelity Mean'].last_valid_index()]
                    data_to_plot_variance[key][row, col] = history[f'{key} T-S Top 1 Fidelity Var'].loc[history[f'{key} T-S Top 1 Fidelity Var'].last_valid_index()]

    for key, data_mean in data_to_plot_mean.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data_mean, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax, vmax=100, vmin=0)

        # Annotate each cell with the standard deviation value
        for i in range(data_mean.shape[0]):
            for j in range(data_mean.shape[1]):
                variance = data_to_plot_variance[key][i, j]
                std_dev = np.sqrt(variance)
                ax.text(j + 0.6, i + 0.6, f"\n{std_dev:.2f}", va='top', ha='right', color='black', fontsize=6)

        ax.set_xticklabels(axes_labels, rotation='vertical', fontsize=15)
        ax.set_yticklabels(axes_labels, rotation='horizontal', fontsize=15)
        ax.set_xlabel('Teacher Training Mechanism', fontsize=15)
        ax.set_ylabel('Student Training Mechanism', fontsize=15)
        plt.savefig(f'images/heatmaps/{loss_name}_{box_pattern}/{key}_{type}.png', dpi=300, bbox_inches='tight')
            

def make_plot(
    histories: List[pd.DataFrame], 
    cols: List[str], 
    title: str, 
    mode: str
) -> None:
    """
    For plots over time. Plots each teacher separately.
    To be called by another function which lists the columns to be plotted - 
    each of these is a subplot.
    
    Args:
        histories: List of dataframes containing historical information for each group of runs.
        cols: List of column names to plot.
        mode: Plotting accuracy, KL or top-1.
    """
    sns.set(style='whitegrid', context='paper', font_scale=1)
    num_groups = len(set([history['Group Name'].iloc[0] for history in histories]))
    n_metrics = len(cols)
    n_cols = min(2, len(cols))
    n_rows = np.ceil(n_metrics / n_cols).astype(int)
    plot_width, plot_height = 13, 3.75*n_rows

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
                line = axs[i].plot(
                    history.index, 
                    history[mean_col], 
                    linewidth=3, 
                    label=group_name, 
                    color=color_dict[group_name], 
                    linestyle=line_styles[line_num%len(line_styles)]
                    )
                axs[i].fill_between(
                    history.index, 
                    history[mean_col] - history[var_col].apply(np.sqrt),
                    history[mean_col] + history[var_col].apply(np.sqrt),
                    alpha=0.2
                    )
                # The key to matching the legend to lines is to check order of dominoes_exp_dict.keys() - these don't correspond to plotting order
                print("Group name: ", group_name, "Line num: ", line_num, "label_group_names: ", label_group_names[line_num])
                if i == 0:  # Only add legend handles once per group (correspond to first subplot)
                    legend_handles.append(
                        mpl.lines.Line2D([0], [0], 
                            color=color_dict[group_name], 
                            linestyle=line_styles[line_num%len(line_styles)], 
                            label=label_group_names[line_num],
                            linewidth=2
                        )
                    )
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
        axs[i].set_xlabel('Training step', fontsize=13)
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles=legend_handles, loc='lower right', ncol=1, bbox_to_anchor=(0.8, 0.05), fontsize=18, frameon=False)
    # fig.suptitle(title, fontsize=20)
    plt.tight_layout(pad=5)
    plt.savefig('images/exhaustivevstime/'+title+'.png', dpi=300, bbox_inches='tight')
    

def make_new_plot(
    histories: List[pd.DataFrame], 
    cols: List[str], 
    title: str, 
    mode: str
) -> None:
    """
    For plots over time. Plots all teachers and all students on a 7x7 grid.
    
    Args:
        histories: List of dataframes containing historical information for each student-teacher pair.
        cols: List of column names to plot.
        mode: Plotting accuracy, KL or top-1.
    """
    fig, axs = plt.subplots(7, 7, figsize=(7.8, 10.5), gridspec_kw={'wspace': 0, 'hspace': 0})
    colors = mpl.cm.get_cmap('viridis', len(cols) + 1)
    line_styles = ['-', '--', '-.', ':']
    common_ylim = (0,100)
    whitespace = 0.1 * (common_ylim[1] - common_ylim[0])  # 10% of the y-range
    adjusted_ylim = (common_ylim[0], common_ylim[1] + whitespace)
    num_ticks = 5  # Adjust as needed for the desired number of ticks
    common_yticks = np.linspace(adjusted_ylim[0], common_ylim[1], num_ticks)
    
    for row, row_name in enumerate(cols):
        for col, col_name in enumerate(cols):
            ax = axs[row][col]
            ax.set_facecolor('white')
            
            # Get the corresponding history dataframe for the student (row) and teacher (col)
            history = next((h for h in histories if h['Group Name'].iloc[0]['S'] == row_name.replace(' Mean','') and h['Group Name'].iloc[0]['T'] == col_name.replace(' Mean','')), None)
        
            for idx, mean_col in enumerate(cols):
                var_col = mean_col.replace(' Mean', ' Var')
                
                if mean_col in history.columns and var_col in history.columns:
                    ax.plot(
                        history.index, 
                        history[mean_col], 
                        linewidth=1,
                        label=mean_col.replace(' Mean', '').replace('_', ' '), 
                        color=colors(idx), 
                        linestyle=line_styles[idx % len(line_styles)]
                    )
                    ax.fill_between(
                        history.index, 
                        history[mean_col] - history[var_col].apply(np.sqrt),
                        history[mean_col] + history[var_col].apply(np.sqrt),
                        color=colors(idx), alpha=0.2
                    )

            ax.set_ylim(adjusted_ylim)

            if col == 0:
                ax.set_yticks(common_yticks)
                ax.set_ylabel(row_name.replace(' Mean', ''), fontsize=10, labelpad=10)
                ax.yaxis.set_tick_params(left=True, labelsize=8)
            else:
                ax.set_yticklabels([])
                ax.set_yticks([])
            
            if row == 6:
                ax.set_xlabel(col_name.replace(' Mean', ''), fontsize=10, labelpad=10)
                ax.set_xticklabels([])
                ax.set_xticks([])
                        
            labelLines(ax.get_lines(), align=False, fontsize=6)

    fig.text(0.5, -0.01, 'Teacher Mechanism', ha='center', va='center', fontsize=13)
    fig.text(-0.01, 0.5, 'Student Mechanism', ha='center', va='center', rotation='vertical', fontsize=13)
    
    if mode == 'acc':
        fig.suptitle('Test accuracy %', fontsize=15)
    elif mode == 'kl':
        fig.suptitle('KL divergence', fontsize=15)
    elif mode == 'fidelity':
        fig.suptitle('Top-1 fidelity %', fontsize=15)

    # Adding legend below the entire plot
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(cols), bbox_to_anchor=(0.5, -0.07))
    
    plt.tight_layout()
    plt.savefig('images/exhaustivevstime/grid/'+title+'.png', dpi=300, bbox_inches='tight')


def counterfactual_plot(
    histories: list[pd.DataFrame], 
    exp_dict: Dict[str, List], 
    title: str
) -> None:
    """For a given run, plot counterfactual test accuracy, KL and top-1 fidelity on different plots.
    
    Args:
        histories: list of edited run histories with an additional 'Group Name' column 
            - tells how data from this history should be used in plotting.
        title: some generic title name to add to plots
        exp_dict: dictionary of experiment names and the associated variable values.
    """
    metric_names = list(exp_dict.keys())
    acc_mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in metric_names]
    kl_mean_cols = [col for col in histories[0].columns if col.replace(' T-S KL Mean', '') in metric_names]
    top1_mean_cols = [col for col in histories[0].columns if col.replace(' T-S Top 1 Fidelity Mean', '') in metric_names]
    acc_mean_cols.sort(key=lambda col: custom_sort(col, 'acc', exp_names))
    kl_mean_cols.sort(key=lambda col: custom_sort(col, 'kl', exp_names))
    top1_mean_cols.sort(key=lambda col: custom_sort(col, 'fidelity', exp_names))

    make_plot(histories, acc_mean_cols, "Counterfactual Test Accuracy"+title)
    make_plot(histories, kl_mean_cols, "Counterfactual T-S KL"+title)
    make_plot(histories, top1_mean_cols, "Counterfactual T-S Top 1 Fidelity"+title)


def wandb_plot(
    histories: List[pd.DataFrame], 
    title: str, 
    grid: bool = True
) -> None:
    """Extract unique column group names and plot them on separate plots - for plotting over training time."""
    mean_cols = [col for col in histories[0].columns if col.replace(' Mean', '') in exp_names]
    mean_cols.sort(key=lambda col: custom_sort(col, 'acc', exp_names))
    
    if grid:
        make_new_plot(histories, mean_cols, title, 'acc')
    else:
        make_new_plot(histories, mean_cols, title, 'acc')


def plot_difference_heatmaps(
    differences: Dict[str, np.ndarray], 
    loss_name: str, 
    box_pattern: str
) -> None:
    """Specifically used to plot heatmaps that are the difference of data with respect to base distillation.
    
    Args:
        differences: keys are the test mechanism and values are the heatmap data.
        loss_name: one of the three 'BASE', 'JACOBIAN', 'CONTRASTIVE'.
        box_pattern: one of 'RANDOM', 'MANDELBROT'.
    """
    for key, data in differences.items():
        fig, ax = plt.subplots()
        heatmap = sns.heatmap(data, cmap='mako', annot=True, fmt=".1f", cbar=True, ax=ax)
        
        ax.set_xticklabels(exp_names, rotation='vertical', fontsize=15)
        ax.set_yticklabels(exp_names, rotation='horizontal', fontsize=15)
        ax.set_xlabel('Teacher Training Mechanism', fontsize=15)
        ax.set_ylabel('Student Training Mechanism', fontsize=15)
        ax.set_title(f'{key.replace("_", " ")} {loss_name}-BASE ACC DIFF')
        plt.savefig(f'images/difference_heatmaps/{loss_name}_{box_pattern}/{key}.png', dpi=300, bbox_inches='tight')


def compute_difference(
    base_hist: List[pd.DataFrame], 
    compare_hist: List[pd.DataFrame], 
    type: str = 'ACC'
) -> Dict[str, np.ndarray]:
    """Difference between final attributes of different loss function to KL distillation loss.
    Retrieves data for calling by plot_difference_heatmaps.
    """
    data_to_diff = {}
    num_keys = len(exp_names)

    # Initialize difference array with zeros and another key for variance with suffix ' Var'
    for key in exp_names:
        data_to_diff[key] = np.zeros((num_keys, num_keys))
        data_to_diff[key + ' Var'] = np.zeros((num_keys, num_keys))
    
    # Create a dictionary for easier lookup based on Group Name for each history in compare_hist
    compare_hist_dict = {tuple(hist['Group Name'].iloc[0].values()): hist for hist in compare_hist}

    # Define a helper function to get the correct column name based on type
    def get_column_name(key, metric):
        match type:
            case 'ACC':
                return f'{key} {metric}'
            case 'KL':
                return f'{key} T-S KL {metric}'
            case 'TOP1':
                return f'{key} T-S Top 1 Fidelity {metric}'
    
    # Compute the differences and variances
    for b_hist in base_hist:
        mechs = b_hist['Group Name'].iloc[0]
        if tuple(mechs.values()) in compare_hist_dict:
            c_hist = compare_hist_dict[tuple(mechs.values())]
            row = exp_names.index(mechs['S'])
            col = exp_names.index(mechs['T'])
            for key in exp_names:
                base_value = b_hist[get_column_name(key, 'Mean')].tail(5).mean()
                compare_value = c_hist[get_column_name(key, 'Mean')].tail(5).mean()
                data_to_diff[key][row, col] = compare_value - base_value
                
                compare_variance_value = c_hist[f"{key} Var"].tail(5).mean()
                data_to_diff[key + ' Var'][row, col] = compare_variance_value

    return data_to_diff


def plot_grid_heatmaps(
    exp_names: List[str],
    loss_names: List[str],
    box_pattern: BoxPatternType,
    wandb_project_name: str,
    groupby_metrics: List[str],
    additional_naming: str,
    exp_row_names: List[str],
    idx: int,
    type: str = 'ACC'
) -> None:
    """
    Plot a grid of heatmaps for each student-teacher mechanism combination, and their variances.
    
    Args:
        exp_names: ordered list of experiment names to plot.
        exp_row_names: ordered list of row names to label the y-axis.
        type: string, 'acc', 'kl' or 'top1'
    """
    print(f"plotting vars in grid heatmap box {box_pattern}")
    num_keys = len(exp_names)
    row_height = 5
    total_height = len(exp_row_names) * row_height
    fig, axes = plt.subplots(nrows=len(exp_row_names), ncols=len(loss_names), figsize=(15, total_height), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    # Colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    vmin = float('inf')
    vmax = -float('inf')

    for loss_index, loss_name in enumerate(loss_names):
        print(f"processing loss {loss_name}")
        
        # Initialize nested dictionaries for this loss_name
        data_to_plot_mean = {key: np.zeros((num_keys, num_keys)) for key in exp_names}
        data_to_plot_variance = {key: np.zeros((num_keys, num_keys)) for key in exp_names}
        
        # Load the correct history
        filename = f"run_data/heatmap {loss_name} {box_pattern}{additional_naming}"
        try:
            with open(filename, "rb") as f: 
                histories = pickle.load(f)
            print('loaded existing data file')
        except:
            histories = heatmap_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)

        for history in histories:
            mechs = history['Group Name'].iloc[0]
            try:
                row = exp_names.index(mechs['S'])
                col = exp_names.index(mechs['T'])
            except: # Mechs not in exp_names
                pass
            
            for key in exp_names:
                match type:
                    case 'ACC':
                        data_to_plot_mean[key][row, col] = history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]
                        data_to_plot_variance[key][row, col] = history[f'{key} Var'].loc[history[f'{key} Var'].last_valid_index()]
                    case 'KL':
                        data_to_plot_mean[key][row, col] = history[f'{key} T-S KL Mean'].loc[history[f'{key} T-S KL Mean'].last_valid_index()]
                        data_to_plot_variance[key][row, col] = history[f'{key} T-S KL Var'].loc[history[f'{key} T-S KL Var'].last_valid_index()]
                    case 'TOP1':
                        data_to_plot_mean[key][row, col] = history[f'{key} T-S Top 1 Fidelity Mean'].loc[history[f'{key} T-S Top 1 Fidelity Mean'].last_valid_index()]
                        data_to_plot_variance[key][row, col] = history[f'{key} T-S Top 1 Fidelity Var'].loc[history[f'{key} T-S Top 1 Fidelity Var'].last_valid_index()]

        for key_index, key in enumerate(exp_row_names):
            ax = axes[key_index][loss_index]
            ax.set_aspect('equal')
            vmin = min(vmin, np.min(data_to_plot_mean[key]))
            vmax = max(vmax, np.max(data_to_plot_mean[key]))

            heatmap = sns.heatmap(data_to_plot_mean[key], cmap='mako', annot=True, fmt=".1f", cbar=True, cbar_ax=cbar_ax, ax=ax, vmax=vmax, vmin=vmin)
            cbar_ax.tick_params(labelsize=15)
            
            # Annotate each cell with the standard deviation value
            for i in range(num_keys):
                for j in range(num_keys):
                    variance = data_to_plot_variance[key][i, j]
                    std_dev = np.sqrt(variance)
                    value = data_to_plot_mean[key][i, j]
                    # Check the brightness of the cell to determine text color
                    norm = plt.Normalize(vmin=vmin, vmax=vmax)
                    rgb = mako_cmap(norm(value))[:3]
                    brightness = sum(rgb) / 3.0
                    text_color = "white" if brightness < 0.5 else "black"
                    ax.text(j + 0.7, i + 0.85, f"{std_dev:.2f}", va='center', ha='center', color=text_color, fontsize=6)

            # Set axis labels
            if key_index == len(exp_row_names) - 1:
                ax.set_xlabel(loss_name, fontsize=15)
            if loss_index == 0:
                ax.set_ylabel(f"Test Mechanism {key}", fontsize=15)
                
            # ax.set_xticklabels(exp_names if key_index == len(exp_row_names) - 1 else [], rotation='vertical', fontsize=10)
            ax.set_xticklabels(exp_names, rotation='vertical', fontsize=10)
            # ax.set_yticklabels(exp_names if loss_index == 0 else [], rotation='horizontal', fontsize=10)
            ax.set_yticklabels(exp_names, rotation='horizontal', fontsize=10)  # Always label yticks

        fig.text(0.5, 0.01, 'Teacher Mechanism', ha='center', fontsize=20)
        fig.text(0.01, 0.5, 'Student Mechanism', va='center', rotation='vertical', fontsize=20)
        plt.tight_layout(rect=[0.02, 0.02, 0.9, 1])
        plt.savefig(f'images/heatmaps/grid_{box_pattern}__{type}_{idx}.png', dpi=300)


def calculate_global_min_max(differences_dict: dict):
    """
    Helper function for normalising grid heatmap values for colourbar.
    
    vmin, vmax clamped +-2 standard deviations.
    """
    all_values = []
    for differences in differences_dict.values():
        for key in differences:
            if isinstance(differences[key], np.ndarray):  # Check if it's an array (ignore variance data)
                all_values.extend(differences[key].flatten())
    mean = np.mean(all_values)
    std_dev = np.std(all_values)
    vmin = mean - 2 * std_dev
    vmax = mean + 2 * std_dev
    return vmin, vmax


def plot_grid_difference_heatmaps(
    exp_names: List[str],
    box_pattern: BoxPatternType,
    exp_row_names: List[str],
    idx: int,
    type: str = 'ACC'
) -> None:
    loss_names = ["Jacobian", "Contrastive"]
    mako_cmap = sns.color_palette('mako', as_cmap=True)

    # Load data from files
    file_names = [F"heatmap {loss_type.value} {box_pattern}" for loss_type in DistillLossType]
    histories = {}
    for file_name in file_names:
        with open(f"run_data/{file_name}", "rb") as f:
            histories[file_name.split()[1]] = pickle.load(f)

    # Calculate differences between specialized distillation losses and base distillation
    base_history = histories["BASE"]
    jacobian_differences = compute_difference(base_history, histories["JACOBIAN"], type)
    contrastive_differences = compute_difference(base_history, histories["CONTRASTIVE"], type)

    differences_dict = {
        "Base": convert_hist_to_dict(base_history, type),  # Absolute values for base distillation
        "Jacobian": jacobian_differences,
        "Contrastive": contrastive_differences
    }

    num_keys = len(exp_names)
    row_height = 5
    total_height = len(exp_row_names) * row_height
    # Remember to add 1 for columns since base plotted by default
    fig, axes = plt.subplots(nrows=len(exp_row_names), ncols=len(loss_names)+1, figsize=(15, total_height), sharex=False, sharey=True)
    fig.subplots_adjust(wspace=0, hspace=0)

    # Normalisation for colourbar
    vmin, vmax = calculate_global_min_max(differences_dict)

    def plot_heatmap_for_loss(ax, loss_name, differences_dict, key, exp_names, vmin, vmax, mako_cmap):
        ax.set_aspect('equal')
        differences = differences_dict[loss_name]
        
        heatmap = sns.heatmap(differences[key], cmap=mako_cmap, annot=True, fmt=".1f", cbar=False, ax=ax, vmax=vmax, vmin=vmin)
        
        # Annotate each cell with the standard deviation value
        variance_data = differences[key + " Var"]
        num_keys = len(exp_names)
        for i in range(num_keys):
            for j in range(num_keys):
                variance = variance_data[i, j]
                std_dev = np.sqrt(variance)
                value = differences[key][i, j]
                
                # Check the brightness of the cell to determine text color
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                rgb = mako_cmap(norm(value))[:3]
                brightness = sum(rgb) / 3.0
                text_color = "white" if brightness < 0.5 else "black"
                ax.text(j + 0.7, i + 0.85, f"{std_dev:.2f}", va='center', ha='center', color=text_color, fontsize=6)
        
        if key_index == len(exp_row_names) - 1: ax.set_xlabel(loss_name, fontsize=15)
        if loss_name == 'Base': ax.set_ylabel(f"Test Mechanism {key}", fontsize=15)
        ax.set_xticklabels(exp_names, rotation='vertical', fontsize=10)
        ax.set_yticklabels(exp_names, rotation='horizontal', fontsize=10)

    for key_index, key in enumerate(exp_row_names):
        ax = axes[key_index][0]
        plot_heatmap_for_loss(ax, "Base", differences_dict, key, exp_names, vmin, vmax, mako_cmap)

    for loss_index, loss_name in enumerate(loss_names):
        print(f"processing loss {loss_name}")
        
        for key_index, key in enumerate(exp_row_names):
            ax = axes[key_index][loss_index + 1]
            plot_heatmap_for_loss(ax, loss_name, differences_dict, key, exp_names, vmin, vmax, mako_cmap)

    # Colourbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=mako_cmap, norm=norm)
    sm.set_array([])  # Set an empty array to link the ScalarMappable with the norm
    fig.colorbar(sm, cax=cbar_ax)

    fig.supxlabel('Teacher Mechanism', fontsize=20)
    fig.supylabel('Student Mechanism', fontsize=20)
    plt.tight_layout(rect=[0.02, 0.02, 0.9, 1])
    plt.savefig(f'images/difference_heatmaps/grid_{box_pattern}_{type}_{idx}.png', dpi=300)


def convert_hist_to_dict(hist: List[pd.DataFrame], type: str = 'ACC') -> Dict[str, np.ndarray]:
    """Converts a list of DataFrames (histories) to a dictionary format."""
    data_dict = {}
    num_keys = len(exp_names)

    # Initialize the dictionary with zeros and another key for variance with suffix ' Var'
    for key in exp_names:
        data_dict[key] = np.zeros((num_keys, num_keys))
        data_dict[key + ' Var'] = np.zeros((num_keys, num_keys))
    
    # Create a dictionary for easier lookup based on Group Name for each history in hist
    hist_dict = {tuple(data['Group Name'].iloc[0].values()): data for data in hist}

    # Helper function to get correct column name based on type
    def get_column_name(key, metric):
        match type:
            case 'ACC':
                return f'{key} {metric}'
            case 'KL':
                return f'{key} T-S KL {metric}'
            case 'TOP1':
                return f'{key} T-S Top 1 Fidelity {metric}'

    # Populate dictionary
    for h_data in hist:
        mechs = h_data['Group Name'].iloc[0]
        if tuple(mechs.values()) in hist_dict:
            current_data = hist_dict[tuple(mechs.values())]
            row = exp_names.index(mechs['S'])
            col = exp_names.index(mechs['T'])
            for key in exp_names:
                value = current_data[get_column_name(key, 'Mean')].tail(5).mean()
                data_dict[key][row, col] = value
                
                variance_value = current_data[f"{key} Var"].tail(5).mean()
                data_dict[key + ' Var'][row, col] = variance_value

    return data_dict


def plot_teachers(
    histories: List[pd.DataFrame], 
    cols: List[str], 
    title: str,
    map_dict: dict[str, str],
) -> None:
    """
    For plots over time. Plots each teacher separately.
    To be called by another function which lists the columns to be plotted - 
    each of these is a subplot.
    
    Args:
        histories: List of dataframes containing historical information for each group of runs.
        cols: List of column names to plot.
        mode: Plotting accuracy, KL or top-1.
        map_dict: Used to change naming convention from Weights and Biases data.
    """
    sns.set(style='whitegrid', context='paper', font_scale=1)
    num_groups = 7
    n_metrics = len(cols)
    n_cols = min(3, n_metrics)
    n_rows = np.ceil(n_metrics / n_cols).astype(int)
    plot_width, plot_height = 11, 3.5*n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(plot_width, plot_height), sharey=True)
    axs = axs.flatten() # Flatten the axs array so that we can iterate over it with a single loop
    if n_metrics < n_rows * n_cols: # Remove unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            fig.delaxes(axs[i])

    colors = mpl.cm.get_cmap('viridis', num_groups+1) # +1 as we ditch yellow
    color_dict = {}
    for i, history in enumerate(histories):
        group_name = history['Group Name'].iloc[0]
        color = colors(i)
        if color[0] > 0.8 and color[1] > 0.8 and color[2] < 0.2:  # Check if i exceeds the maximum index of colors
            continue
        if group_name not in color_dict:
            color_dict[group_name] = colors(i)  # Use the next color in the colormap

    for ax in axs:
        ax.set_prop_cycle(color=[color_dict[group_name] for group_name in color_dict])
        for spine in ax.spines.values(): # Solid border
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')
    legend_handles = []

    for i, mean_col in enumerate(cols): # Iterate over subplots
        ax = axs[i]
        xvals = []  # List to store x-coordinates for labels
        for line_num, history in enumerate(histories): # Iterate over lines
            group_name = history['Group Name'].iloc[0]
            if mean_col in history.columns:
                index_position = history[mean_col].last_valid_index()
                x_position = np.random.uniform(0.1 * index_position, index_position)
                xvals.append(x_position)
                line = ax.plot(
                        history.index, 
                        history[mean_col], 
                        linewidth=3, 
                        label=map_dict[group_name], 
                        color=color_dict[group_name], 
                    )
                # The key to matching the legend to lines is to check order of dominoes_exp_dict.keys() - these don't correspond to plotting order
                print(f"Group name: {group_name} Line num {line_num} Label group names {label_group_names[line_num]}")
                if i == 0:  # Only add legend handles once per group (correspond to first subplot)
                    legend_handles.append(
                        mpl.lines.Line2D(
                            [0], 
                            [0], 
                            color=color_dict[group_name], 
                            label=label_group_names[line_num],
                            linewidth=2
                        )
                    )
        ax.set_title(f"Test Mechanism {map_dict[mean_col]}", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.set_ylim(-5, 105)
        labelLines(ax.get_lines(), fontsize=13, align=False, xvals=xvals)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.supxlabel("Training Evaluation Step", fontsize=15)
    fig.supylabel('Test Accuracy', fontsize=15)
    leg = fig.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), fontsize=15, title="Teacher Mechanism", handlelength=1, ncol=len(labels), title_fontsize=15)
    for legobj in leg.legendHandles: legobj.set_linewidth(3.0)
    plt.tight_layout()
    plt.savefig('images/exhaustivevstime/'+title+'.png', dpi=500, bbox_inches='tight')


if __name__ == "__main__":
    """Somewhat immutable variables"""
    exp_names = [config.name for config in ConfigGroups.exhaustive]
    label_group_names = ["IAB", "IA", "IB", "AB", "B", "A", "I"] # Actual order of names here
    loss_names = [loss_type.value for loss_type in DistillLossType]
    
    """Mutable"""
    box_pattern = 'RANDOM'
    model_name = 'RN18AP'
    dataset_type = 'DOMINOES'
    config_type = 'EXHAUSTIVE'
    additional_naming = '' # For names appended to end of typical project naming convention (ALPHA, BETA etc)
    wandb_project_name = f"DISTILL-{model_name}-{dataset_type}-{config_type}-{box_pattern}{additional_naming}"

    # 0 for heatmap, 1 for plots, 2 for grid plots (all teachers on one plot), 3 for diff heatmaps, 4 for grid of heatmaps with variance, 5 for diff grid heatmaps with variance
    mode = 5
    groupby_metrics = ["experiment.name", "experiment_s.name"] # IMPORTANT: teacher mechanism must go first in the groupby_metrics list

    if mode == 0:
        for loss_name in loss_names:
            type = 'acc'
            print(f'calculating heatmap mode 0 for {wandb_project_name} {loss_name} {type}')
            try:
                filename = f"run_data/heatmap {loss_name} {box_pattern}{additional_naming}"
                with open(filename, "rb") as f: histories = pickle.load(f)
                print('loaded existing data file')
            except:
                histories: List[pd.DataFrame] = heatmap_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)

            # plot_counterfactual_heatmaps(histories, exp_names, loss_name, box_pattern, type=type, additional_naming=additional_naming)
            plot_mean_variance_heatmaps(histories, exp_names, loss_name, box_pattern)
            
    elif mode == 1:
        for t_exp_name in exp_names:
            for loss_name in loss_names:
                title = f"{model_name} {t_exp_name} {loss_name}"
                try: # Files already calculated and exist
                    filename = f"run_data/vstime {t_exp_name} {loss_name}"
                    with open(filename, "rb") as f: histories = pickle.load(f)
                except:
                    histories = wandb_get_data(project_name=wandb_project_name, t_exp_name=t_exp_name, loss_name=loss_name, model_name=model_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, grid=False, plot_tmechs_together=False)

                wandb_plot(histories, title, grid=False)
    
    elif mode == 2:
        for loss_name in loss_names:
            title = f"{model_name} {loss_name}"
            try: # Files already calculated and exist
                filename = f"run_data/vstime {loss_name} grid"
                with open(filename, "rb") as f: histories = pickle.load(f)
            except: 
                histories = wandb_get_data(project_name=wandb_project_name, t_exp_name=None, loss_name=loss_name, model_name=model_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, grid=True)

            wandb_plot(histories, title, grid=True)

    elif mode == 3:
        box_pattern = "MANDELBROT"
        file_names = [F"heatmap {loss_type.value} {box_pattern}" for loss_type in DistillLossType]
        histories = {}
        for file_name in file_names:
            with open(f"run_data/{file_name}", "rb") as f:
                histories[file_name.split()[1]] = pickle.load(f)

        # Calculate differences between specialized distillation losses and base distillation
        base_history = histories["BASE"]
        jacobian_differences = compute_difference(base_history, histories["JACOBIAN"])
        contrastive_differences = compute_difference(base_history, histories["CONTRASTIVE"])

        plot_difference_heatmaps(jacobian_differences, "JACOBIAN", box_pattern)
        plot_difference_heatmaps(contrastive_differences, "CONTRASTIVE", box_pattern)
    
    elif mode == 4:
        types = ["ACC", "KL", "TOP1"]
        box_pattern = "MANDELBROT"
        mako_cmap = sns.color_palette("mako", as_cmap=True)  # Get the 'mako' colormap from Seaborn
        exp_row_names_list = [['I', 'A', 'B'], ['AB', 'IB', 'IA', 'IAB']]
        for i, exp_row_names in enumerate(exp_row_names_list):
            for type in types:
                plot_grid_heatmaps(exp_names, loss_names, box_pattern, wandb_project_name, groupby_metrics, additional_naming, exp_row_names, (i+1), type)
    
    elif mode == 5:
        types = ["ACC", "KL", "TOP1"]
        box_pattern = "RANDOM"
        exp_row_names_list = [['I', 'A', 'B'], ['AB', 'IB', 'IA', 'IAB']]
        for i, exp_row_names in enumerate(exp_row_names_list):
            for type in types:
                plot_grid_difference_heatmaps(exp_names, box_pattern, exp_row_names, (i+1), type)
    
    elif mode == 6:
        """Teachers."""
        if box_pattern == "RANDOM":
            teacher_name_mapping_wandb = {"C": "I", "B": "A", "M": "B", "CB": "IA", "MB": "AB", "CM": "IB", "CMB": "IAB"}
            wandb_project_name = f"{model_name} {dataset_type} {config_type}"
        elif box_pattern == "MANDELBROT":
            teacher_name_mapping_wandb = {"I": "I", "A": "A", "B": "B", "IA": "IA", "AB": "AB", "IB": "IB", "IAB": "IAB"}
            wandb_project_name = f"{model_name} {dataset_type} {config_type} {box_pattern}"
        cols = list(teacher_name_mapping_wandb.keys())
        runs = api.runs(wandb_project_name)
        histories = []
        for run in runs:
            history = run.history()
            history['Group Name'] = run.config.get("experiment", {}).get("name")
            histories.append(history)
        
        plot_teachers(histories, cols=cols, title=f"teacher_{box_pattern}", map_dict=teacher_name_mapping_wandb)


