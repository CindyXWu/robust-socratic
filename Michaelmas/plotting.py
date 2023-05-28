import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Optional
import numpy as np
from labellines import labelLine, labelLines


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


def plot_acc(train_acc: List[float], test_acc: List[float], it: int, base_name: Optional[str] = '', title: Optional[str] = ''):
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


def plot_df(df: pd.DataFrame, base_name: Optional[str] = '', title: Optional[str] = ''):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    
    df_melt = df.reset_index().melt('index', var_name='a', value_name='acc')
    df_melt[['Type', 'Fraction']] = df_melt.a.str.split("_", expand=True)

    line_styles = ['-', '--']  # define line styles for 'Type'
    type_styles = {typ: style for typ, style in zip(df_melt['Type'].unique(), line_styles)}

    # Use colormap to assign colors based on unique fractions
    unique_fracs = df_melt['Fraction'].unique()
    cmap = plt.cm.get_cmap('cool', len(unique_fracs))
    frac_colors = {frac: cmap(i) for i, frac in enumerate(unique_fracs)}

    lines = []
    for (frac, typ), data in df_melt.groupby(['Fraction', 'Type']):
        data = data.dropna()  # Filter out rows with NaN values
        line, = ax.plot(data["index"], data["acc"], linestyle=type_styles[typ], color=frac_colors[frac], label=str(frac))
        lines.append(line)

    labelLines(lines, align=False, fontsize=11)

    # Remove box around the graph (except bottom and left axes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for 'Type'
    custom_lines = [plt.Line2D([0], [0], color='black', linestyle=style) for style in line_styles]
    ax.legend(custom_lines, df_melt['Type'].unique(), loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=False, shadow=False, ncol=len(df.columns), frameon=False)

    plt.ylabel('Accuracy')
    plt.xlabel('Iteration/100')
    plt.ylim([20, 110])
    plt.grid(False)
    plt.tight_layout()
    
    if base_name != '':
        fig.savefig(base_name + title + '.png', bbox_inches='tight')
    else:
        plt.show()

    plt.close("all")


def heatmap_df(dataset_names: List[str], student):
    # Initialize an empty list to store all dataframes
    all_data = []

    # Load each dataset, process it and append to all_data
    for dataset_name in dataset_names:
        # Load the dataset
        df = pd.read_csv(f'Michaelmas/{student}_student_results/'+ dataset_name + '.csv')  # replace this line with your actual data loading code
        # Melt the DataFrame from wide format to long format
        df_melt = df.reset_index().melt('index', var_name='a', value_name='accuracy')
        # Split the 'a' column into 'Type', 'Fraction'
        df_melt[['Type', 'Fraction']] = df_melt['a'].str.split('_', expand=True)
        # Add a 'Dataset' column
        df_melt['Dataset'] = dataset_name
        # Drop the 'a' column
        df_melt = df_melt.drop(columns=['a'])
        # Drop NaNs
        df_melt = df_melt.dropna()
        # Convert Fraction to float
        df_melt['Fraction'] = df_melt['Fraction'].astype(float)
        # Append the processed dataframe to all_data
        all_data.append(df_melt)
    
    print(df_melt.head())
    # Concatenate all dataframes in all_data into a single dataframe
    df_all = pd.concat(all_data, ignore_index=True)

    # Create a heatmap for each dataset
    for dataset_name in dataset_names:
        # Filter the dataframe for the current dataset
        df_dataset = df_all[df_all['Dataset'] == dataset_name]
        # Pivot the dataframe to create a matrix suitable for a heatmap
        df_pivot = df_dataset.pivot_table(index='Type', columns='Fraction', values='accuracy', aggfunc='mean')

        plt.figure(figsize=(5, 2))  # Adjust the figure size
        sns.set(font_scale=1)  # Increase the font size
        sns.heatmap(df_pivot, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Accuracy'}, linewidths=.5)
        # plt.title(f'Accuracy for {dataset_name}')
        plt.savefig(f'Michaelmas/images/heatmaps/{student}/{dataset_name}.png', bbox_inches='tight')


def heatmap_diff_df(dataset_names: List[str], student):
    """Student teacher difference heatmap."""
    # Initialize an empty list to store all dataframes
    all_data = []

    # Load each dataset, process it and append to all_data
    for dataset_name in dataset_names:
        # Load the student dataset
        df_student = pd.read_csv(f'Michaelmas/{student}_student_results/' + dataset_name + '.csv')
        # Load the teacher dataset
        df_teacher = pd.read_csv('Michaelmas/teacher_results/'+ dataset_name + '.csv')
        # Melt the DataFrames from wide format to long format
        df_student_melt = df_student.reset_index().melt('index', var_name='a', value_name='accuracy')
        df_teacher_melt = df_teacher.reset_index().melt('index', var_name='a', value_name='accuracy')
        # Split the 'a' column into 'Type', 'Fraction'
        df_student_melt[['Type', 'Fraction']] = df_student_melt['a'].str.split('_', expand=True)
        df_teacher_melt[['Type', 'Fraction']] = df_teacher_melt['a'].str.split('_', expand=True)
        print(df_teacher_melt.head())
        print(df_student_melt.head())
        # Calculate the accuracy difference between student and teacher
        df_diff = df_student_melt.copy()
        df_diff['accuracy'] = df_student_melt['accuracy'] - df_teacher_melt['accuracy']
        # Add a 'Dataset' column
        df_diff['Dataset'] = dataset_name
        # Drop the 'a' column
        df_diff = df_diff.drop(columns=['a'])
        # Drop NaNs
        df_diff = df_diff.dropna()
        # Convert Fraction to float
        df_diff['Fraction'] = df_diff['Fraction'].astype(float)
        # Append the processed dataframe to all_data
        all_data.append(df_diff)

    # Concatenate all dataframes in all_data into a single dataframe
    df_all = pd.concat(all_data, ignore_index=True)

    # Create a heatmap for each dataset
    for dataset_name in dataset_names:
        # Filter the dataframe for the current dataset
        df_dataset = df_all[df_all['Dataset'] == dataset_name]
        # Pivot the dataframe to create a matrix suitable for a heatmap
        df_pivot = df_dataset.pivot_table(index='Type', columns='Fraction', values='accuracy', aggfunc='mean')

        plt.figure(figsize=(5, 2))  # Adjust the figure size
        sns.set(font_scale=1)  # Increase the font size
        sns.heatmap(df_pivot, cmap='YlGnBu', annot=True, fmt=".1f", cbar_kws={'label': 'Accuracy Difference'}, linewidths=.5)
        plt.savefig(f'Michaelmas/images/diffheatmaps/{student}/{dataset_name}_diff.png', bbox_inches='tight')


def plot_data(data_arr, d1, d2, title=''):
    """"Plot function that plots two features against each other."""
    class1_data = data_arr[data_arr[:, -1] == 0]
    class2_data = data_arr[data_arr[:, -1] == 1]
    sns.set_style('white')
    palette = sns.color_palette("Set2")
    fig, ax = plt.subplots(figsize=(4, 4))
    # Plot scatter for class 1 and class 2
    scatter1 = ax.scatter(class1_data[:, d1], class1_data[:, d2], c=[palette[0]], alpha=0.7, edgecolors='none', s=50)
    scatter2 = ax.scatter(class2_data[:, d1], class2_data[:, d2], c=[palette[1]], alpha=0.7, edgecolors='none', s=50)
    ax.set_xlabel('$x_2$', fontsize=18)
    ax.set_ylabel('$x_3$', fontsize=18)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # Remove the box around the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.grid(False)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    # Maintain aspect ratio
    ax.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig('data '+title+'.png')


if __name__ == '__main__':

    fig_type = 1 # 0: plot data, 1: plot heatmap, 2: plot heatmap difference
    base_dir = 'Michaelmas/large_student_results/'

    dataset_names = []
    # List of complex indices (cols) to randomise (see utils.py)
    X_list = ['[1 2]', '[1]', '[2]']
    # List of test complex indices (cols) to randomise
    SC_list = ['[0]', '[0 1]', '[0 2]']
    for X in X_list:
        for SC in SC_list:
            filename = f'train_{X}_test_{SC}'
            dataset_names.append(filename)

    match fig_type:
        case 0:
            for dataset in dataset_names:
                df = pd.read_csv(base_dir+dataset+'.csv', index_col=0)
                print(df.head())
                plot_df(df, base_name=base_dir, title=dataset)
        case 1:
            heatmap_df(dataset_names, student='large')
        case 2:
            heatmap_diff_df(dataset_names, student='large')
        