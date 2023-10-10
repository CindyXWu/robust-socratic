from collections import defaultdict
import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging
import wandb
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines
import seaborn as sns
from typing import List, Dict

from config_setup import ConfigGroups, BoxPatternType, DistillLossType
from plotting_common import drop_non_numeric_columns, clean_history, smooth_history, get_grouped_runs, custom_sort, recursive_namespace, create_histories_list, condition_for_neither, condition_for_similarity, condition_for_student, condition_for_teacher, get_nested_value, save_df_csv

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)

# sns.set_style("whitegrid")
# sns.set_palette("pastel")
mpl.use('Agg')


def heatmap_get_data(project_name: str,
                     loss_name: str,
                     box_pattern: str,
                     groupby_metrics: List[str],
                     additional_naming: str) -> List[pd.DataFrame]:
    """Get data from wandb for experiment set, filter and group. 
    Calculate mean/var of metrics and return a list of history dataframes with mean/var interleaved.
    
    Args:
        config: DistillConfig object containing experiment settings.
        additional_naming: Additional string to add to the end of the file name.
    """
    runs = api.runs(project_name)
        
    filtered_runs = []
    min_step = 1200 # Filter partially logged/unfinished runs

    # Filter for loss and correct experiment name, and remove crashed/incomplete runs
    for run in runs:
        if run.config.get('distill_loss_type') == loss_name and run.config.get("experiment", {}).get("name") in label_group_names:
            try:
                history = run.history()
            except:
                history = run.history
            if '_step' in history.columns and history['_step'].max() >= min_step:
                history = drop_non_numeric_columns(history) # Remove artifacts
                history = clean_history(history) # Remove NaNs
                #history = smooth_history(history)
                run.history  = history
                filtered_runs.append(run)
                
    # Check list of runs is not empty
    assert(len(filtered_runs) > 0), "No runs found with the given settings"
    
    """Key of form: tuple of groupby metrics (in order it's passed in, in groupby_metrics)"""
    grouped_runs: Dict[str, List[Run]] = get_grouped_runs(filtered_runs, groupby_metrics)
    
    # Compute means/var for all metrics for each group of runs
    histories: List[pd.DataFrame] = create_histories_list(grouped_runs, mode='exhaustive')
    
    file_name = f"run_data/heatmap {loss_name} {box_pattern}{additional_naming}"
    with open(file_name, "wb") as f:
        pickle.dump(histories, f)
        
    return histories


def aggregate_conditions(combined_history: list[pd.DataFrame], type: str) -> Dict[str, np.ndarray]:
    """Get grouped data for heatmap plots of distribution shift experiments."""

    # Initialize matrices for means and variances
    aggregated_means = {'similarity': 0, 'student': 0, 'teacher': 0, 'neither': 0, 'other': 0}
    aggregated_vars = {'similarity': 0, 'student': 0, 'teacher': 0, 'neither': 0, 'other': 0}
    condition_counts = {'similarity': 0, 'student': 0, 'teacher': 0, 'neither': 0, 'other': 0}

    for history in combined_history:
        student = history['Group Name'].iloc[0]['S']
        teacher = history['Group Name'].iloc[0]['T']

        for key in exp_names:
            if condition_for_similarity(student, teacher, key):
                condition_name = 'similarity'
            elif condition_for_student(student, teacher, key):
                condition_name = 'student'
            elif condition_for_teacher(student, teacher, key):
                condition_name = 'teacher'
            elif condition_for_neither(student, teacher):
                condition_name = 'neither'
            else:
                condition_name = 'other'
                """Includes cases where the teacher and student are the same, but the key is not in either of them."""

            match type:
                case 'acc':
                    aggregated_means[condition_name] += history[f'{key} Mean'].loc[history[f'{key} Mean'].last_valid_index()]
                    aggregated_vars[condition_name] += history[f'{key} Var'].loc[history[f'{key} Var'].last_valid_index()]
                case 'kl':
                    aggregated_means[condition_name] += history[f'{key} T-S KL Mean'].loc[history[f'{key} T-S KL Mean'].last_valid_index()]
                    aggregated_vars[condition_name] += history[f'{key} T-S KL Var'].loc[history[f'{key} T-S KL Var'].last_valid_index()]
                case 'top1':
                    aggregated_means[condition_name] += history[f'{key} T-S Top 1 Fidelity Mean'].loc[history[f'{key} T-S Top 1 Fidelity Mean'].last_valid_index()]
                    aggregated_vars[condition_name] += history[f'{key} T-S Top 1 Fidelity Var'].loc[history[f'{key} T-S Top 1 Fidelity Var'].last_valid_index()]

            condition_counts[condition_name] += 1

    for condition in aggregated_means:
        aggregated_means[condition] /= condition_counts[condition]
        aggregated_vars[condition] /= condition_counts[condition]

    return {
        'means': aggregated_means,
        'variances': aggregated_vars
    }


def plot_aggregated_data(aggregated_data: Dict[str, np.ndarray], loss_name: str, box_pattern: str, type: str) -> None:
    """
    HEATMAP VERSION
    Plotting function for data produced by data retrieval function aggregate_conditions.
    """
    conditions = list(aggregated_data['means'].keys())
    
    # Create a dataframe for seaborn
    df_means = pd.DataFrame([aggregated_data['means']])
    df_vars = pd.DataFrame([aggregated_data['variances']])

    # Plotting means heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_means, annot=True, fmt=".1f", cmap="mako", yticklabels=["Mean"])
    plt.title("Aggregated Means")
    plt.xticks(range(len(conditions)), conditions)
    plt.savefig(f"images/aggregated_heatmaps/{loss_name}_{box_pattern}_{type}_means.png", dpi=300, bbox_inches="tight")

    # Plotting variances heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_vars, annot=True, fmt=".1f", cmap="mako", yticklabels=["Variance"])
    plt.title("Aggregated Variances")
    plt.xticks(range(len(conditions)), conditions)
    plt.savefig(f"images/aggregated_heatmaps/{loss_name}_{box_pattern}_{type}_variances.png", dpi=300, bbox_inches="tight")
    
## For violin plots ======================================================================================

def violinplot_get_data(project_name: str,
                        loss_name: str,
                        box_pattern: str,
                        groupby_metrics: List[str],
                        additional_naming: str) -> List[pd.DataFrame]:
    runs = api.runs(project_name)

    filtered_runs = []
    filtered_histories = []
    min_step = 1200  # Filter partially logged/unfinished runs

    # Filter for loss and correct experiment name, and remove crashed/incomplete runs
    for run in tqdm(runs):
        try:
            history = run.history()
        except:
            history = run.history
        if ('_step' in history and history['_step'].max() > min_step and 
            run.config.get('distill_loss_type') == loss_name and 
            run.config.get("experiment", {}).get("name") in label_group_names):
            key = tuple([get_nested_value(run.config, m) for m in groupby_metrics])
            if '_step' in history.columns and history['_step'].max() >= min_step:
                history = drop_non_numeric_columns(history)  # Remove artifacts
                history = clean_history(history)  # Remove NaNs
                #history = smooth_history(history)
                history['Group Name'] = history.apply(lambda row: {'T': key[0], 'S': key[1]}, axis=1)
                run.history = history
                filtered_runs.append(run)
                filtered_histories.append(history)
    assert(len(filtered_runs) > 0), "No runs found with the given settings"

    file_name = f"run_data/aggregated violin {loss_name} {box_pattern} {additional_naming}"
    with open(file_name, "wb") as f:
        pickle.dump(filtered_histories, f)

    return filtered_histories

def aggregate_conditions_violin(combined_history: List[pd.DataFrame], type: str) -> pd.DataFrame:
    all_data = []
    condition_mapping = {'similarity': [], 'student': [], 'teacher': [], 'neither': [], 'other': []}
    for history in tqdm(combined_history):
        student = history['Group Name'].iloc[0]['S']
        teacher = history['Group Name'].iloc[0]['T']
        print(history.head())
        for key in exp_names:
            if condition_for_similarity(student, teacher, key):
                condition_name = 'similarity'
            elif condition_for_student(student, teacher, key):
                condition_name = 'student'
            elif condition_for_teacher(student, teacher, key):
                condition_name = 'teacher'
            elif condition_for_neither(student, teacher):
                condition_name = 'neither'
            else:
                condition_name = 'other'

            if type == 'ACC':
                metric_name = f'{key}'
            elif type == 'KL':
                metric_name = f'{key} T-S KL'
            elif type == 'TOP-1':
                metric_name = f'{key} T-S Top 1 Fidelity'
            else:
                raise ValueError("Unsupported type")

            value = history[metric_name].iloc[-1]
            
            data_dict = {
                'Value': value,
                'Condition': condition_name,
                'Key': key,
                'Teacher': teacher,
                'Student': student
            }
            all_data.append(data_dict)

            # assert value < 50, f"WARNING: {metric_name} for {teacher} {student} on {key} is {value}"
            condition_mapping[condition_name].append(f"{teacher} {student}")

    final_df = pd.DataFrame(all_data)
    save_df_csv(final_df, title=f'final_{type}_df')
    return final_df, condition_mapping



def plot_aggregated_data_scatter(df: pd.DataFrame, loss_name: str, box_pattern: str, type: str, title: str, make_legend: bool) -> None:
    # Define a fixed order for the conditions
    condition_order = ['similarity', 'student', 'teacher', 'neither', 'other']

    # Set style and context suitable for an academic paper
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=2.5)  # Increase the font scale

    # Increase the figure size
    plt.figure(figsize=(12, 7.5))
    ax = sns.stripplot(x="Condition", y="Value", hue="Key", data=df, order=condition_order,
                       jitter=True, dodge=True, palette="Paired", marker="o", edgecolor="gray", size=10)
    y_axes_names = {"ACC": "Student Test Accuracy", "KL": "Teacher-Student KL Divergence", "TOP-1": "Teacher-Student Top-1 Fidelity"}
    plt.ylabel(y_axes_names[type], fontsize=24, labelpad=15)
    plt.xlabel("Teacher/Distillation/Test Mechanisms Condition", fontsize=24, labelpad=30)
    if not make_legend:
        ax.legend_.remove()
    if make_legend:
        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles=handles, labels=labels, title="Test Mech", bbox_to_anchor=(0.2, 0.95), loc=2, borderaxespad=0., fontsize=17, title_fontsize=24, frameon=False)
        for legobj in leg.legendHandles:
            legobj.set_sizes([200])
            legobj.set_edgecolor('black')
            legobj.set_facecolor(legobj.get_facecolor())
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"images/aggregated_scatter/{loss_name}_{box_pattern}_{type}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Somewhat immutable things
    exp_names = [config.name for config in ConfigGroups.exhaustive]
    label_group_names = ["IAB", "IA", "IB", "AB", "B", "A", "I"] # Actual order of names here
    loss_names = [loss_type.value for loss_type in DistillLossType]
    
    box_pattern = 'MANDELBROT'
    model_name = 'RN18AP'
    dataset_type = 'DOMINOES'
    config_type = 'EXHAUSTIVE'
    additional_naming = '' # For names appended to end of typical project naming convention
    wandb_project_name = f"DISTILL-{model_name}-{dataset_type}-{config_type}-{box_pattern}{additional_naming}"

    # 0 for combined heatmaps, 1 for combined violin plots, 2 for combined scatter plots
    mode = 2
    groupby_metrics = ["experiment.name", "experiment_s.name"]
    
    if mode == 0:
        box_pattern = "MANDELBROT"
        for loss_name in loss_names:
            type = 'ACC'
            print(f'calculating aggregated heatmap mode 4 for {wandb_project_name} {box_pattern} {loss_name} {type}')
            try:
                filename = f"run_data/heatmap {loss_name} {box_pattern} {additional_naming}"
                with open(filename, "rb") as f: histories = pickle.load(f)
            except:
                # IMPORTANT: teacher mechanism must go first in groupby_metrics list
                histories: List[pd.DataFrame] = heatmap_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)

            aggregated_data = aggregate_conditions(combined_history=histories, type=type)
            df_means, df_vars = pd.DataFrame([aggregated_data['means']]), pd.DataFrame([aggregated_data['variances']])
            df_means.to_csv(f"run_data/{loss_name}_{box_pattern}_aggregated_means.csv", index=False)
            df_vars.to_csv(f"run_data/{loss_name}_{box_pattern}_aggregated_vars.csv", index=False)

            plot_aggregated_data(aggregated_data, loss_name, box_pattern, type=type)
    
    elif mode == 1:
        box_pattern = "MANDELBROT"
        for loss_name in loss_names:
            type = 'ACC'
            title = f"{type} {loss_name} {box_pattern}{additional_naming}"
            filename = f"run_data/aggregated violin {loss_name} {box_pattern}{additional_naming}"

            print(f'calculating aggregated violin mode 1 for {wandb_project_name} {title}')

            try:
                with open(filename, "rb") as f:
                    violin_histories = pickle.load(f)
                save_df_csv(violin_histories[0], 'violin histories after opening saved raw data')
            except:
                # IMPORTANT: teacher mechanism must go first in groupby_metrics list
                violin_histories: List[pd.DataFrame] = violinplot_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)

            violin_df, condition_map = aggregate_conditions_violin(violin_histories, type=type)
            # Now, directly use violin_df for plotting since it's already in the right format
            plot_aggregated_data_violin(violin_df, loss_name, box_pattern, type=type, title=title)
    
    elif mode == 2:
        box_pattern = "RANDOM"
        type = 'KL'
        for loss_name in loss_names:
            make_legend = True if loss_name == 'CONTRASTIVE' else False
            title = f"{type} {loss_name} {box_pattern}{additional_naming}"
            wandb_project_name = f"DISTILL-{model_name}-{dataset_type}-{config_type}-{box_pattern}{additional_naming}"
            filename = f"run_data/aggregated violin {loss_name} {box_pattern}{additional_naming}"
            print(f'calculating aggregated scatter mode 2 for {wandb_project_name} {title}')

            try:
                with open(filename, "rb") as f:
                    violin_histories = pickle.load(f)
                save_df_csv(violin_histories[0], 'scatterplot histories after opening saved raw data')
            except:
                # IMPORTANT: teacher mechanism must go first in groupby_metrics list
                violin_histories: List[pd.DataFrame] = violinplot_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)

            violin_df, condition_map = aggregate_conditions_violin(violin_histories, type=type)
            # Now, directly use violin_df for plotting since it's already in the right format
            plot_aggregated_data_scatter(violin_df, loss_name, box_pattern, type=type, title=title, make_legend=make_legend)

            print(pd.DataFrame(dict([(k, pd.Series(v)) for k, v in condition_map.items()])) )