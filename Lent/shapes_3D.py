
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# methods for sampling unconditionally/conditionally on a given factor
def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
    factors: np array shape [6,batch_size].
                factors[i]=factors[i,:] takes integer values in 
                range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
    indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        print(factors[factor])
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


def sample_random_batch(batch_size):
    """ Samples a random batch of images.
    Args:
    batch_size: number of images to sample.

    Returns:
    batch: images shape [batch_size,64,64,3].
    """
    indices = np.random.choice(n_samples, batch_size)
    ims = []
    for ind in indices:
        im = images[ind]
        im = np.asarray(im)
        ims.append(im)
    ims = np.stack(ims, axis=0)
    ims = ims / 255. # normalise values to range [0,1]
    ims = ims.astype(np.float32)
    return ims.reshape([batch_size, 64, 64, 3])


def sample_batch(batch_size, fixed_factor, fixed_factor_value):
    """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
        the other factors varying randomly.
    Args:
    batch_size: number of images to sample.
    fixed_factor: index of factor that is fixed in range(6).
    fixed_factor_value: integer value of factor that is fixed 
        in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

    Returns:
    batch: images shape [batch_size,64,64,3]
    """
    factors = np.zeros([len(_FACTORS_IN_ORDER), batch_size],
                        dtype=np.int32)
    for factor, name in enumerate(_FACTORS_IN_ORDER):
        num_choices = _NUM_VALUES_PER_FACTOR[name]
        factors[factor] = np.random.choice(num_choices, batch_size)
    factors[fixed_factor] = fixed_factor_value
    indices = get_index(factors)
    ims = []
    for ind in indices:
        im = images[ind]
        im = np.asarray(im)
        ims.append(im)
    ims = np.stack(ims, axis=0)
    ims = ims / 255. # normalise values to range [0,1]
    ims = ims.astype(np.float32)
    return ims.reshape([batch_size, 64, 64, 3])


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
    plt.show()

class Shapes3D(Dataset):
    def __init__(self):
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
        
        # Convert into 12 scalar labels
        self.new_labels = np.empty([self.n_samples])
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

def dataloader_3D_shapes(load_type, batch_size):
    """Load dataset."""
    dataset = Shapes3D()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(load_type=='train'), drop_last=True)
    return dataloader

if __name__ == "__main__":

    # load dataset
    dataset = h5py.File('3dshapes.h5', 'r')
    # print(dataset.keys())
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                        'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                            'scale': 8, 'shape': 4, 'orientation': 15}
    
    shapes_dataloader = dataloader_3D_shapes('train', 1)
    # Get a single batch from the dataloader
    iterator = iter(shapes_dataloader)
    batch = next(iterator)
    images, labels = batch
    print(images.shape)
    print(labels.shape)
    # show_images_grid(images, 16)
    print(labels)