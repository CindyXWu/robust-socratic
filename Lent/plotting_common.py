from collections import defaultdict
import pandas as pd
import numpy as np
import os
import einops
import torch
import warnings
from torch.utils.data import DataLoader
from labellines import labelLines
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import savgol_filter
import seaborn as sns
import wandb
from wandb.sdk.wandb_run import Run
from sklearn.manifold import TSNE


os.chdir(os.path.dirname(os.path.abspath(__file__)))
image_dir = "images/"
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    
    
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
        plt.savefig(f'{image_dir}{title}.png')
    else:
        plt.show()
        
        
def plot_saliency_map(saliency: torch.Tensor, title: str) -> None:
    plt.imshow(saliency.cpu(), cmap=plt.cm.hot)
    plt.axis('off')
    plt.title(title)
    plt.savefig(f'{image_dir}/{title}.png')


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