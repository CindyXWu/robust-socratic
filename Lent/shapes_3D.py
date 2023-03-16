
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Shapes3D(Dataset):
    def __init__(self, randomise_shape=False, randomise_colour=False):
        """Args:
        fixed_factors: factors indexing into _FACTORS_IN_ORDER that are predictive of label (currently only supports a single factor). Index of factor that is fixed in range(6).
        """
        # load dataset
        self.dataset = h5py.File('3dshapes.h5', 'r')
        self.images = self.dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        self.labels = self.dataset['labels']  # array shape [480000,6], float64
        self.image_shape = self.images.shape[1:]  # [64,64,3]
        self.label_shape = self.labels.shape[1:]  # [6]
        self.n_samples = self.labels.shape[0]  # 10*10*10*8*4*15=480000

        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                            'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                                'scale': 8, 'shape': 4, 'orientation': 15}
        
        self.randomise_shape = randomise_shape
        self.randomise_colour = randomise_colour

        # Convert into 12 scalar labels
        self.new_labels = np.empty([self.n_samples])

        # if self.randomise_shape:
            
        # if self.randomise_colour:

        # else:
        for idx in range(self.n_samples):
            shape = self.labels[idx,4]
            # Split object colour into 3 distinct groups
            if self.labels[idx,2] < 0.33:
                hue = 0
            elif 0.33 <= self.labels[idx,3] < 0.67:
                hue = 1
            else:
                hue = 2
            self.new_labels[idx] = hue * 4 + shape

        # Get labels as one hot encodings for mixup
        self.one_hot_encode(12)

    def one_hot_encode(self, num_classes):
        """
        Convert numerical class labels into one-hot encodings.
        Args:
            labels (torch.Tensor): tensor of numerical class labels
            num_classes (int): total number of classes
        Returns:
            one_hot (torch.Tensor): tensor of one-hot encodings
        """
        self.oh_labels = F.one_hot(torch.tensor(self.new_labels).to(torch.int64), num_classes=num_classes)

    def __getitem__(self, idx):
        """Returns:
        image: numpy array image of shape [64,64,3]
        label: torch tensor label of shape [12] (one hot encoding)
        """
        return self.images[idx,:,:,:], self.oh_labels[idx, :]

    def __len__(self):
        return self.n_samples

def show_images_grid(imgs_, num_images=25):
  ncols = int(np.ceil(num_images**0.5))
  nrows = int(np.ceil(num_images / ncols))
  _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
  axes = axes.flatten()

  for ax_i, ax in enumerate(axes):
    if ax_i < num_images:
      ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')
     
def dataloader_3D_shapes(load_type, batch_size):
    """Load dataset."""
    dataset = Shapes3D()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(load_type=='train'), drop_last=True)
    return dataloader

if __name__ == "__main__":
    shapes_dataloader = dataloader_3D_shapes('train', 1)
    # Get a single batch from the dataloader
    iterator = iter(shapes_dataloader)
    batch = next(iterator)
    images, labels = batch
    print(images.shape)
    print(labels.shape)
    # show_images_grid(images, 16)
    print(labels)