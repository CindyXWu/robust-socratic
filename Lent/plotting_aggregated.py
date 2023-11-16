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
from plotting_common import (
    drop_non_numeric_columns, 
    clean_history, 
    smooth_history, 
    get_grouped_runs, 
    custom_sort, 
    recursive_namespace, 
    create_histories_list, 
    condition_neither, 
    condition_similarity, 
    condition_student,
    condition_teacher,
    condition_key_equals_t,
    condition_key_equals_s,
    condition_overlap_not_in_key,
    get_nested_value,
    save_df_csv
)

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
            if condition_similarity(student, teacher, key):
                condition_name = 'similarity'
            elif condition_student(student, teacher, key):
                condition_name = 'student'
            elif condition_teacher(student, teacher, key):
                condition_name = 'teacher'
            elif condition_neither(student, teacher):
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


## For violin/scatter plots ===========================================================================


def violinplot_get_data(project_name: str,
                        loss_name: str,
                        box_pattern: str,
                        additional_naming: str) -> List[pd.DataFrame]:
    """Get data from Weights and Biases in format suitable for teacher mechanism agnostic grouping.
    
    Valid for violin and scatter plot by logic of grouping.
    """
    runs = api.runs(project_name)
    filtered_runs = []
    filtered_histories = []
    min_step = 1200

    for run in tqdm(runs):  # Filter for loss and correct experiment name, and remove crashed/incomplete runs
        try:
            history = run.history()
        except:
            history = run.history
        if ('_step' in history and history['_step'].max() > min_step and 
            run.config.get('distill_loss_type') == loss_name and 
            run.config.get("experiment", {}).get("name") in label_group_names):
            if '_step' in history.columns and history['_step'].max() >= min_step:
                history = drop_non_numeric_columns(history)  # Remove artifacts
                history = clean_history(history)  # Remove NaNs
                history['Group Name'] = history.apply(lambda row: {'T': key[0], 'S': key[1]}, axis=1)
                run.history = history
                filtered_runs.append(run)
                filtered_histories.append(history)
    assert(len(filtered_runs) > 0), "No runs found with the given settings"

    file_name = f"run_data/aggregated violin {loss_name} {box_pattern} {additional_naming}"
    with open(file_name, "wb") as f:
        pickle.dump(filtered_histories, f)

    return filtered_histories


def aggregate_conditions(combined_history: List[pd.DataFrame], type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take list of history dataframes. Go through every single history and group into a particular condition. 
    Append final value of metric given by type to column named 'value'.
    Append the teacher and student mechanisms to list of teacher-student pairs in condition dictionary. Create a new dataframe from this and return as condition map.
    Create a dataframe with the columns 'Value', 'Condition', 'Key', 'Teacher', 'Student', then concat (since they all have same cols).
    """
    all_data = []
    condition_map = {'Similarity': [], 'Student': [], 'Teacher': [], 'Student TP': [], 'Teacher SP':[], 'Overlap': [], 'Neither': [], 'Other': []}

    for history in tqdm(combined_history):
        student = history['Group Name'].iloc[0]['S']
        teacher = history['Group Name'].iloc[0]['T']

        for key in exp_names:
            if condition_similarity(student, teacher, key):
                condition_name = 'Similarity'
            elif condition_student(student, teacher, key):
                condition_name = 'Student'
            elif condition_teacher(student, teacher, key):
                condition_name = 'Teacher'
            elif condition_key_equals_s(student, teacher, key):
                condition_name = 'Student TP'
            elif condition_key_equals_t(student, teacher, key):
                condition_name = 'Teacher SP'
            elif condition_overlap_not_in_key(student, teacher, key):
                condition_name = 'Overlap'
            elif condition_neither(student, teacher):
                condition_name = 'Neither'
            else:
                condition_name = 'Other'

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

            condition_map[condition_name].append(f"{teacher} {student} {key}")

    df_condition_map = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in condition_map.items()]))
    final_df = pd.DataFrame(all_data)
    save_df_csv(final_df, title=f'final_{type}_df')
    save_df_csv(df_condition_map, title=f'condition_mapping_{type}_df')

    return final_df, condition_map


def plot_aggregated_data_scatter(all_data: dict[str, pd.DataFrame], box_pattern: str, type: str) -> None:     
    """Plot strip scatter plots for all loss functions in a single row."""
    # Define a fixed order for the conditions
    condition_order = ['Similarity', 'Student', 'Teacher', 'Student TP', 'Teacher SP', 'Overlap', 'Neither', 'Other']

    sns.set(style="white", context="paper", font_scale=1.3)

    num_plots = len(all_data)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), sharey=True)

    if num_plots == 1: 
        axes = [axes]

    y_axes_names = {"ACC": "Student Test Accuracy", "KL": "KL Divergence", "TOP-1": "Top-1 Fidelity"}
    all_handles, all_labels = set(), []  # To store unique handles and labels

    for idx, (loss_name, df) in enumerate(all_data.items()):
        ax = axes[idx]
        sns.stripplot(x="Condition", y="Value", hue="Key", data=df, order=condition_order, jitter=False, dodge=True, palette="Paired", marker="o", edgecolor="gray", size=6, ax=ax)

        # Update axis labels and titles
        if idx == 0: ax.set_ylabel(y_axes_names[type], fontsize=15, labelpad=15)
        ax.set_xlabel("Dataset Mechanisms Condition", fontsize=15, labelpad=15)
        ax.set_title(f"{loss_name.capitalize()} Loss", fontsize=18)

        # Collect handles and labels from each subplot
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in all_labels:  # Avoid duplicates
                all_handles.add(handle)
                all_labels.append(label)

        ax.legend_.remove()

    fig.legend(all_handles, all_labels, loc='lower center', ncol=len(all_labels), bbox_to_anchor=(0.5, 0), fontsize=17, title="Test Mechanism", title_fontsize=17)

    plt.tight_layout()
    plt.savefig(f"images/aggregated_scatter/{box_pattern}_{type}.png", dpi=500, bbox_inches="tight")



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

    # 0 for combined heatmaps, 1 for scatter plots
    mode = 1
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
        box_pattern = "RANDOM"
        type = 'TOP-1'
        data_to_plot = {}   # Holds data that goes into plotting scatter

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
            except: # IMPORTANT: teacher mechanism must go first in groupby_metrics list
                violin_histories: List[pd.DataFrame] = violinplot_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, additional_naming=additional_naming)

            violin_df, condition_map = aggregate_conditions(violin_histories, type=type)
            data_to_plot[loss_name] = violin_df
        
        plot_aggregated_data_scatter(data_to_plot, box_pattern, type=type)