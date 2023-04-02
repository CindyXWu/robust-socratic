
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
import einops
import timeit

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Shapes3D(Dataset):
    def __init__(self, randomise=False, mechanisms=None):
        """Args:
            mechanisms: list indexing into _FACTORS_IN_ORDER that is predictive of label. Index of factor that is fixed in range(6).
        Labels returned one hot encoded.
        """
        with h5py.File('3dshapes.h5', 'r') as dataset:  # use 'with' statement to ensure the file is closed
            self.images = np.array(dataset['images'])  # convert h5py Dataset to numpy array shape [480000,64,64,3], uint8 in range(256)
            self.labels = np.array(dataset['labels'])  # convert h5py Dataset to numpy array shape [480000,6], float64
            self.image_shape = self.images.shape[1:]  # [64,64,3]
            self.label_shape = self.labels.shape[1:]  # [6]
            self.n_samples = self.labels.shape[0]  # 10*10*10*8*4*15=480000
        self._FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                            'orientation']
        self._NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                                'scale': 8, 'shape': 4, 'orientation': 15}
        self.randomise = randomise
        self.mechanisms = mechanisms

        # Convert into scalar labels
        self.new_labels = np.empty([self.n_samples])
        self.num_classes = 12

        shape = self.labels[:,4]
        hue = np.zeros_like(shape)
        hue[np.logical_and(0.33 <= self.labels[:,2], self.labels[:,2] < 0.67)] = 1
        hue[self.labels[:, 2] >= 0.67] = 2
        self.new_labels = hue*4 + shape

        if mechanisms and len(mechanisms)==1 and not randomise:
            mechanism = mechanisms[0]
            match mechanism:
                case 1:
                    wall_hue = self.labels[:,1]
                    # Filter images and labels based on wall hue matching new_labels
                    mask = np.array([True if int(wall_hue[i]*10) == self.new_labels[i]%10 else False for i in range(self.n_samples)])
                case 0:
                    floor_hue = self.labels[:,0]
                    mask = np.array([True if int(floor_hue[i]*10) == self.new_labels[i]%10 else False for i in range(self.n_samples)])
                case 3:
                    scale = self.labels[:,3]
                    # Hash maps to easy comparison - now idx contains values 0-7 for each image's scale
                    # Didn't need to do for other cases because their values were nice numbers
                    _, idx = np.unique(scale, return_inverse=True)
                    # idx[i] is hashed scale value for image i
                    mask = np.array([True if idx[i] == self.new_labels[i]%8 else False for i in range(self.n_samples)])
            self.images = self.images[mask]
            self.new_labels = self.new_labels[mask]
            self.n_samples = self.new_labels.shape[0]

        # Get labels as one hot encodings for mixup
        self.one_hot_encode()

    def one_hot_encode(self):
        """
        Convert numerical class labels into one-hot encodings.
        Args:
            labels (torch.Tensor): tensor of numerical class labels
            num_classes (int): total number of classes
        Returns:
            one_hot (torch.Tensor): tensor of one-hot encodings
        """
        self.oh_labels = F.one_hot(torch.tensor(self.new_labels).to(torch.int64), num_classes=self.num_classes)

    def __getitem__(self, idx):
        """Returns:
        image: numpy array image of shape [3, 64, 64]
        label: scalar torch tensor label in range 1-12
        """
        # Rearrange format using einops
        x = einops.rearrange(self.images[idx,:,:,:], 'h w c -> c h w')
        x = torch.from_numpy(x).to(torch.float32)
        y = int(self.new_labels[idx])
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return self.n_samples

def show_images_grid(imgs_, num_images=4):
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
      ax.set_xticks([])
      ax.set_yticks([])
    else:
      ax.axis('off')
     
def dataloader_3D_shapes(load_type, batch_size, randomise=False, mechanisms=None):
    """Load dataset."""
    dataset = Shapes3D(randomise=randomise, mechanisms=mechanisms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(load_type=='train'), drop_last=True, num_workers=0)
    return dataloader

if __name__ == "__main__":
    bsize = 4
    shapes_dataloader = dataloader_3D_shapes('train', bsize, mechanisms=[0])
    for images, labels in shapes_dataloader:
        print(labels)
        images = einops.rearrange(images, 'b c h w -> b h w c')
        show_images_grid(images, bsize)
        break