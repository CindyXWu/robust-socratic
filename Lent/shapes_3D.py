
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
    def __init__(self, randomise=False, mechanisms=[]):
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
        self.mechanisms = mechanisms # Possibilities: 0, 3

        # Convert into scalar labels
        self.new_labels = np.empty([self.n_samples])
        self.num_classes = 8

        # Make baseline labels - classes 1-8 for shape/colour
        if not randomise:
            shape = self.labels[:,4]
            hue = np.zeros_like(shape)
            hue[self.labels[:, 2] >= 0.5] = 1
            self.new_labels = hue*4 + shape
        else:
            self.new_labels = np.random.randint(0, self.num_classes, size=self.n_samples)
        
        # I could apply data mask inside if statement for each mechanism - if num mechanisms large, creates boilerplate code
        # Use 'logical and' to combine masks instead
        if 0 in mechanisms:
            floor_hue = self.labels[:,0]
            mask_0 = np.array([True if (floor_hue[i]*10).astype(int) == self.new_labels[i] else False for i in range(self.n_samples)])
        else:
            # Identity mask
            mask_0 = np.ones(self.n_samples, dtype=bool)
        if 3 in mechanisms:
            scale = self.labels[:,3]
            # Hash maps to integers - idx[i] is scale value for image i
            _, idx = np.unique(scale, return_inverse=True)
            mask_3 = np.array([True if idx[i] == self.new_labels[i] else False for i in range(self.n_samples)])
        else:
            mask_3 = np.ones(self.n_samples, dtype=bool)
        mask = np.logical_and(mask_0, mask_3)
        self.images = self.images[mask]
        self.new_labels = self.new_labels[mask]
        self.n_samples = self.new_labels.shape[0]

        # Get labels as one hot encodings for mixup
        self.one_hot_encode()

    def one_hot_encode(self):
        """Convert numerical class labels into one-hot encodings."""
        self.oh_labels = F.one_hot(torch.tensor(self.new_labels).to(torch.int64), num_classes=self.num_classes)

    def __getitem__(self, idx):
        """Returns:
            image: numpy array image of shape [3, 64, 64]
            label: scalar torch tensor label in range 1-12
        """
        x, y = self.process_img(self.images[idx,:,:,:], self.new_labels[idx])
        return x, y

    def process_img(self, x, y):
        """Convert from numpy array uint8 to torch float32 tensor. Convert y to torch long tensor."""
        x = einops.rearrange(x, 'h w c -> c h w')
        x = torch.div(torch.from_numpy(x).to(torch.float32),255)
        y = torch.tensor(y).to(torch.long)
        return x, y
        
    def __len__(self):
        return self.n_samples

def show_images_grid(imgs_, class_labels, num_images):
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
    plt.show()
     
def dataloader_3D_shapes(load_type, batch_size, randomise=False, mechanisms=[]):
    """Load dataset."""
    dataset = Shapes3D(randomise=randomise, mechanisms=mechanisms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(load_type=='train'), drop_last=True, num_workers=0)
    return dataloader

if __name__ == "__main__":
    bsize = 32
    shapes_dataloader = dataloader_3D_shapes('train', bsize, mechanisms=[0, 3])
    for images, labels in shapes_dataloader:
        print(labels)
        images = einops.rearrange(images, 'b c h w -> b h w c')
        show_images_grid(images, labels, bsize)
        break