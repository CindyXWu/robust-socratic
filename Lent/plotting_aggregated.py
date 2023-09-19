from collections import defaultdict
import pandas as pd
import numpy as np
import os
import yaml
import pickle
import logging
import wandb

import matplotlib.pyplot as plt
import matplotlib as mpl
from labellines import labelLines
import seaborn as sns
from typing import List, Dict

from config_setup import ConfigGroups, BoxPatternType, DistillLossType
from plotting_common import drop_non_numeric_columns, clean_history, smooth_history, get_grouped_runs, custom_sort, recursive_namespace, create_histories_list, condition_for_neither, condition_for_similarity, condition_for_student, condition_for_teacher

os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

api = wandb.Api(overrides=None, timeout=None, api_key =None)

# sns.set_style("whitegrid")
# sns.set_palette("pastel")


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
    min_step = 300 # Filter partially logged/unfinished runs

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
    histories = create_histories_list(grouped_runs, mode='exhaustive')
    
    file_name = f"run_data/heatmap {loss_name} {box_pattern} {additional_naming}"
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

# def violinplot_get_data(project_name: str,
#                         loss_name: str,
#                         box_pattern: str,
#                         groupby_metrics: List[str],
#                         additional_naming: str) -> pd.DataFrame:
#     runs = api.runs(project_name)
        
#     filtered_runs = []
#     min_step = 300 # Filter partially logged/unfinished runs

#     # Filter for loss and correct experiment name, and remove crashed/incomplete runs
#     for run in runs:
#         if run.config.get('distill_loss_type') == loss_name and run.config.get("experiment", {}).get("name") in label_group_names:
#             history = run.history()
#             if '_step' in history.columns and history['_step'].max() >= min_step:
#                 history = drop_non_numeric_columns(history) # Remove artifacts
#                 history = clean_history(history) # Remove NaNs
#                 #history = smooth_history(history)
#                 run.history  = history
#                 filtered_runs.append(run)
                
#     # Check list of runs is not empty
#     assert(len(filtered_runs) > 0), "No runs found with the given settings"
    
#     """Key of form: tuple of groupby metrics (in order it's passed in, in groupby_metrics)"""
#     grouped_runs: Dict = get_grouped_runs(filtered_runs, groupby_metrics)
#     combined_histories = [run.history for group in grouped_runs.values() for run in group]
    
#     # Compute means/var for all metrics for each group of runs
#     df = prepare_data_for_violinplot(combined_histories)
    
#     file_name = f"run_data/aggregated violin {loss_name} {box_pattern} {additional_naming}"
#     with open(file_name, "wb") as f:
#         pickle.dump(df, f)
        
#     return df
    

# def aggregate_conditions_violin(combined_history: list[pd.DataFrame], type: str) -> pd.DataFrame:
#     """Prepare data in a format suitable for seaborn's violinplot based on conditions."""

#     all_data = []

#     for history in combined_history:
#         student = history['Group Name'].iloc[0]['S']
#         teacher = history['Group Name'].iloc[0]['T']

#         for key in exp_names:
#             if condition_for_similarity(student, teacher, key):
#                 condition_name = 'similarity'
#             elif condition_for_student(student, teacher, key):
#                 condition_name = 'student'
#             elif condition_for_teacher(student, teacher, key):
#                 condition_name = 'teacher'
#             elif condition_for_neither(student, teacher):
#                 condition_name = 'neither'
#             else:
#                 condition_name = 'other'

#             if type == 'acc':
#                 metric_name = f'{key} Mean'
#             elif type == 'kl':
#                 metric_name = f'{key} T-S KL Mean'
#             elif type == 'top1':
#                 metric_name = f'{key} T-S Top 1 Fidelity Mean'
#             else:
#                 raise ValueError("Unsupported type")

#             temp_df = history[[metric_name]].copy()
#             temp_df.rename(columns={metric_name: 'Value'}, inplace=True)
#             temp_df['Metric'] = key
#             temp_df['Condition'] = condition_name
#             all_data.append(temp_df)

#     final_df = pd.concat(all_data, axis=0)
#     return final_df
    

# def plot_aggregated_data_violin(aggregated_data: Dict[str, np.ndarray], loss_name: str, box_pattern: str, type: str) -> None:
#     """
#     VIOLIN PLOT VERSION
#     Plotting function for data produced by data retrieval function aggregate_conditions.
#     """
#     conditions = list(aggregated_data['means'].keys())
    
#     # Create a dataframe for seaborn in long format
#     df_means = pd.DataFrame({
#         'condition': conditions,
#         'value': list(aggregated_data['means'].values()),
#         'type': 'Mean'
#     })
#     df_vars = pd.DataFrame({
#         'condition': conditions,
#         'value': list(aggregated_data['variances'].values()),
#         'type': 'Variance'
#     })
    
#     # Concatenate both dataframes
#     df = pd.concat([df_means, df_vars], axis=0)
    
#     # Plotting violin plot
#     plt.figure(figsize=(12, 6))
#     sns.violinplot(x="condition", y="value", hue="type", data=df, split=True, inner="quart")
#     plt.title("Aggregated Data")
#     plt.xticks(range(len(conditions)), conditions)
#     plt.savefig(f"images/aggregated_violin/{loss_name}_{box_pattern}_{type}.png", dpi=300, bbox_inches="tight")


# def prepare_data_for_violinplot(grouped_runs: Dict[tuple, List]) -> pd.DataFrame:
#     """Prepare data in a format suitable for seaborn's violinplot."""
#     all_data = []
#     for key, runs in grouped_runs.items():
#         for run in runs:
#             history = run.history
            
#             for metric in history.columns:
#                 temp_df = history[[metric]].copy()
#                 temp_df.rename(columns={metric: 'Value'}, inplace=True)
#                 temp_df['Metric'] = metric
#                 temp_df['Group Name'] = f"T: {key[0]}, S: {key[1]}"
#                 all_data.append(temp_df)
    
#     final_df = pd.concat(all_data, axis=0)
#     return final_df

def violinplot_get_data(project_name: str,
                        loss_name: str,
                        box_pattern: str,
                        groupby_metrics: List[str],
                        additional_naming: str) -> pd.DataFrame:
    runs = api.runs(project_name)

    filtered_runs = []
    min_step = 300  # Filter partially logged/unfinished runs

    # Filter for loss and correct experiment name, and remove crashed/incomplete runs
    for run in runs:
        if run.config.get('distill_loss_type') == loss_name and run.config.get("experiment", {}).get("name") in label_group_names:
            history = run.history()
            if '_step' in history.columns and history['_step'].max() >= min_step:
                history = drop_non_numeric_columns(history)  # Remove artifacts
                history = clean_history(history)  # Remove NaNs
                #history = smooth_history(history)
                run.history = history
                filtered_runs.append(run)

    # Check list of runs is not empty
    assert(len(filtered_runs) > 0), "No runs found with the given settings"

    # Key of form: tuple of groupby metrics (in order it's passed in, in groupby_metrics)
    grouped_runs: Dict = get_grouped_runs(filtered_runs, groupby_metrics)

    combined_histories = []
    for group, runs in grouped_runs.items():
        for run in runs:
            history = run.history
            # Add Group Name column
            history['Group Name'] = f"T: {group[0]}, S: {group[1]}"
            combined_histories.append(history)
            
    # Compute means/var for all metrics for each group of runs and aggregate conditions
    df = aggregate_conditions_violin(combined_histories, type="acc")  # You can change type based on need

    file_name = f"run_data/aggregated violin {loss_name} {box_pattern} {additional_naming}"
    with open(file_name, "wb") as f:
        pickle.dump(df, f)

    return df


def aggregate_conditions_violin(combined_history: List[pd.DataFrame], type: str) -> pd.DataFrame:
    all_data = []

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

            if type == 'acc':
                metric_name = f'{key} Mean'
            elif type == 'kl':
                metric_name = f'{key} T-S KL Mean'
            elif type == 'top1':
                metric_name = f'{key} T-S Top 1 Fidelity Mean'
            else:
                raise ValueError("Unsupported type")

            temp_df = history[[metric_name]].copy()
            temp_df.rename(columns={metric_name: 'Value'}, inplace=True)
            temp_df['Metric'] = key
            temp_df['Condition'] = condition_name
            all_data.append(temp_df)

    final_df = pd.concat(all_data, axis=0)
    return final_df


def plot_aggregated_data_violin(df: pd.DataFrame, loss_name: str, box_pattern: str, type: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Metric", y="Value", hue="Condition", data=df, split=True, inner="quart")
    plt.title("Aggregated Data")
    plt.savefig(f"images/aggregated_violin/{loss_name}_{box_pattern}_{type}.png", dpi=300, bbox_inches="tight")


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

    # 0 for combined heatmaps, 1 for combined violin plots
    mode = 1
    groupby_metrics = ["experiment.name", "experiment_s.name"]
    
    if mode == 0:
        box_pattern = "MANDELBROT"
        for loss_name in loss_names:
            type = 'acc'
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
            type = 'acc'
            print(f'calculating aggregated violin mode 5 for {wandb_project_name} {box_pattern} {loss_name} {type}')
            filename = f"run_data/aggregated violin {loss_name} {box_pattern} {additional_naming}"
            try:
                with open(filename, "rb") as f:
                    violin_df = pickle.load(f)
            except:
                # IMPORTANT: teacher mechanism must go first in groupby_metrics list
                violin_df: pd.DataFrame = violinplot_get_data(project_name=wandb_project_name, loss_name=loss_name, box_pattern=box_pattern, groupby_metrics=groupby_metrics, additional_naming=additional_naming)
                with open(filename, "wb") as f:
                    pickle.dump(violin_df, f)
            
            # Now, directly use violin_df for plotting since it's already in the right format
            plot_aggregated_data_violin(violin_df, loss_name, box_pattern, type=type)