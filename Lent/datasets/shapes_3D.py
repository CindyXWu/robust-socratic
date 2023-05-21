
from matplotlib import pyplot as plt
import numpy as np
import h5py
import os
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
import einops
import sys
from numpy.typing import NDArray
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from plotting import *

# Get the current script's directory
script_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(script_dir, '../data/3dshapes.h5')

class Shapes3D(Dataset):
    def __init__(self, randomise=False, floor_frac=0, scale_frac=0):
        """
        A combination of object hue and shape is used to create 8 classes as base label.
        It is not possible to actually get rid of the mechanisms here. Instead, we will fully randomise to remove them.
        Args:
            randomise: whether to randomise the 'semantically important' feature (a combination of object hue, shape)
            floor_frac: fraction of images where floor hue is predictive of label. Set to 0 is equivalent to randomising or removing the feature.
            scale_frac: fraction of images where scale is predictive of label. Set to 0 is equivalent to randomising or removing the feature.
        """
        with h5py.File(file_path, 'r') as dataset:
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
        self.floor_frac = floor_frac
        self.scale_frac = scale_frac
        # should keep False for now - other method doesn't work well yet
        self.remove_mech = False

        # Convert into scalar labels
        self.new_labels = np.empty([self.n_samples])
        self.classes = 8

        # Make baseline labels - classes 1-8 for shape/colour
        if not randomise:
            shape = self.labels[:,4]
            hue = np.zeros_like(shape)
            hue[self.labels[:, 2] >= 0.5] = 1
            self.new_labels = hue*4 + shape
        else:
            self.new_labels = np.random.randint(0, self.classes, size=self.n_samples)
        
        # Add mechanisms at given fraction
        mask_0 = np.ones_like(self.new_labels, dtype=bool)
        mask_3 = np.ones_like(self.new_labels, dtype=bool)
        if floor_frac != 0:
            floor_hue = self.labels[:,0]
            # Images where floor hue is the same as the label
            mask_0 = np.array([True if (floor_hue[i]*10).astype(int) == self.new_labels[i] else False for i in range(self.n_samples)])
            if not self.remove_mech:
                mask_0 = self.apply_frac_randomised(mask_0, floor_frac)
        if scale_frac != 0:
            scale = self.labels[:,3]
            # Hash maps to integers - idx[i] is scale value for image i
            _, idx = np.unique(scale, return_inverse=True)
            mask_3 = np.array([True if idx[i] == self.new_labels[i] else False for i in range(self.n_samples)])
            mask_3 = self.apply_frac_randomised(mask_3, scale_frac)
        mask = np.logical_and(mask_0, mask_3)
        self.images = self.images[mask]
        self.new_labels = self.new_labels[mask]
        self.n_samples = self.new_labels.shape[0]

        # Get labels as one hot encodings for mixup
        self.one_hot_encode()

    def apply_frac_randomised(self, mask: NDArray[np.bool_], frac: float):
        """In case where frac is not 1, randomise the mechanism for frac of the images.
        Assume a frac of 0.6 means only 60% of images have the mechanism corresponding to label, and 40% have a randomised mechanism.
        This is not applied as augmentation but rather as a dataset preprocessing step.
        An alternative definition is given below.
        """
        numels = np.sum(mask) # Number of spurious images
        # Calculate number of images we want with this mechanism randomised
        rand_numels = int(numels*(1-frac)/frac)
        false_indices = np.where(mask == False)[0]
        select_idx = np.random.choice(false_indices, rand_numels, replace=False)
        # Add images with randomised mechanism to mask
        mask[select_idx] = True
        return mask

    def get_index(self, factors: NDArray[np.int_]):
        """ Converts factors to indices in range(num_data)
        Args:
            factors: np array shape [6,batch_size].
                    factors[i]=factors[i,:] with values in 
                    range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

        Returns:
            indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices

    def one_hot_encode(self):
        """Convert numerical class labels into one-hot encodings."""
        self.oh_labels = F.one_hot(torch.tensor(self.new_labels).to(torch.int64), num_classes=self.classes)

    def __getitem__(self, idx):
        """Returns:
            image: numpy array image of shape [3, 64, 64]
            label: scalar torch tensor label in range 1-12
        """
        x = self.images[idx,:,:,:]
        if self.remove_mech:
            x = self.remove_floor_hue(x)
        x, y = self.process_img(x, self.new_labels[idx])
        return x, y

    def remove_floor_hue(self, x: NDArray[np.uint8]):
        """Augmentation for cue correlation fraction by changing floor hue to a random colour.
        Args:
            tolerance: tolerance for Euclidian distance between pixels and floor colour.
        """
        # Select random uniform value between 0 and 1
        rand = np.random.uniform()
        # If floor hue is same as object or wall there is no way I can edit its value separately
        # Return and skip this image
        if self.pixel_similarity(x[63,0,:], x[31, 42,:]) or self.pixel_similarity(x[63,0,:], x[31, 0,:]):
            return x
        # If rand > floor_frac then remove floor hue mechanism by randomising pixels
        if 0 < self.floor_frac < 1 and rand > self.floor_frac:
            floor_col = x[63,0,:]
            mask = self.pixel_similarity(x, floor_col, tolerance=3)
            x[mask] = np.random.randint(0, 256, size=3)
        return x

    def pixel_similarity(self, arr_1, arr_2, tolerance=10):
        abs_diff = np.abs(arr_1 - arr_2)
        mask = np.all(abs_diff <= tolerance, axis=-1)
        return mask
    
    def process_img(self, x, y):
        """Convert from numpy array uint8 to torch float32 tensor. Convert y to torch long tensor."""
        x = einops.rearrange(x, 'h w c -> c h w')
        x = torch.div(torch.from_numpy(x).to(torch.float32),255)
        y = torch.tensor(y).to(torch.long)
        return x, y
        
    def __len__(self):
        return self.n_samples

     
def dataloader_3D_shapes(load_type, batch_size, randomise=False, floor_frac=0, scale_frac=0):
    """Load dataset."""
    dataset = Shapes3D(randomise=randomise, floor_frac=floor_frac, scale_frac=scale_frac)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(load_type=='train'), drop_last=True, num_workers=0)
    return dataloader


if __name__ == "__main__":
    bsize = 32
    shapes_dataloader = dataloader_3D_shapes('train', bsize, floor_frac=0.5, scale_frac=0.5)
    for images, labels in shapes_dataloader:
        print(labels)
        images = einops.rearrange(images, 'b c h w -> b h w c')
        show_images_grid(images, labels, bsize)
        break