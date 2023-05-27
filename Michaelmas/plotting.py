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
    cmap = plt.cm.get_cmap('winter', len(unique_fracs))
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
    base_dir = 'Michaelmas/teacher_results/'
    filename = 'train_[1 2]_test_[0]'
    df = pd.read_csv(base_dir+filename+'.csv', index_col=0)
    plot_df(df, base_name=base_dir, title=filename)