"""Dataset creation for box and cue datasets. Includes own version of image batch plotting."""
import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100
from torch.utils.data import DataLoader, Dataset

import os
import einops
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Type, Union, Dict, Tuple, List

from config_setup import BoxPatternType

## Uncomment out below lines for reproducibility
# import torch.backends.cudnn as cudnn
# torch.manual_seed(0)
# cudnn.deterministic = True
# cudnn.benchmark = False


def get_box_mask(
    mask: torch.Tensor,
    loc: int,
    box_size: int,
    box_color: Union[float, torch.Tensor],
    w: int,
    h: int,
    num_classes: int) -> torch.Tensor:
    """Box placement for num_classes = 10 or 100.
    
    Args:
        mask: Default Torch tensor of zeros with same shape as image.
        box_color: Single color if uniform or pattern in torch Tensor form.
    Returns:
        mask: Torch tensor of zeros with box added.
    """
    if (loc == 0): # upper left
        mask[:, :box_size, :box_size] = box_color
    elif (loc == 1): # upper center
        mask[:, :box_size, (h - box_size) // 2 : (h + box_size) // 2] = box_color
    elif (loc == 2): # upper right
        mask[:, :box_size, -box_size:] = box_color
        
    elif (loc == 3): # middle left
        mask[:, (w - box_size) // 2 : (w + box_size) // 2, :box_size] = box_color
    elif (loc == 4): # middle center
        mask[:, (w - box_size) // 2 : (w + box_size) // 2, (h - box_size) // 2 : (h + box_size) // 2] = box_color
    elif (loc == 5): # middle right
        mask[:, (w - box_size) // 2 : (w + box_size) // 2, -box_size:] = box_color
        
    elif (loc == 6): # lower left
        mask[:, -box_size:, :box_size] = box_color
    elif (loc == 7): # lower center
        mask[:, -box_size:, (h - box_size) // 2 : (h + box_size) // 2] = box_color
    elif (loc == 8): # lower right
        mask[:, -box_size:, -box_size:] = box_color
        
    elif (loc == 9):
        if num_classes == 10:
            # draw nothing if CIFAR-10
            pass
        else:
            # mid-upper-left if CIFAR-100
            mask[:, (w - box_size) // 2 - (w // 4) : 
                    (w + box_size) // 2 - (w // 4), 
                    (h - box_size) // 2 - (h // 4) : 
                    (h + box_size) // 2 - (h // 4)] = box_color

    return mask


class boxDataset(Dataset):
    """Box perturbation dataset"""
    def __init__(self, 
                 dataset: Union[CIFAR10, CIFAR100], 
                 image_frac: float = 1.0,
                 cue_frac: float = 1.0,
                 randomize_img: bool = False,
                 randomize_cue: bool = False):
        self.dataset = dataset
        self.classes = dataset.classes
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)
        self.called_mixup = False # Flag for preventing infinite recursion

        self.image_frac, self.cue_frac = image_frac, cue_frac
        self.randomize_img, self.randomize_cue = randomize_img, randomize_cue
        
        self.cue_ids: Dict = get_cue_ids(labels=self.targets, n_classes=self.n_classes, cue_frac=cue_frac)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]

        if self.randomize_img:
            image = self.dataset[np.random.randint(0, len(self.dataset))][0]
        if self.image_frac < 1 and np.random.uniform() > self.image_frac:
            image = torch.zeros_like(image)

        # Box cues
        put_cue_attribute = (np.random.uniform() < self.cue_frac) if self.cue_ids is None else self.cue_ids[label][item]
        
        if put_cue_attribute:
            mask = self.get_box(torch.zeros_like(image), label)
            # Zero image where mask is present and add mask
            image = mask + (mask == 0).all(axis=0) * image

        return image, label
        
    def get_box(self, mask: torch.Tensor, label: int) -> torch.Tensor:
        """Box creation function.
        
        Args:
            mask: Default Torch tensor of zeros with same shape as image.
            label: Of image.
        
        Returns:
            mask: Torch tensor of zeros with box added.
        """
        loc = np.random.randint(0, 10) if self.randomize_cue else (label % 10)

        l = (label // 10) if self.n_classes == 100 else 10   # only use color for CIFAR-100
        color: float = np.random.uniform() if (self.randomize_cue and self.n_classes == 100) else (l / 10) # only use color for CIFAR-100

        # HSV -> RGB -> BGR -> add w,h dimensions
        rgb = matplotlib.colors.hsv_to_rgb([color, 1, 255])[[2, 0, 1]][..., None, None]

        box_size = 3
        box_color = torch.Tensor((rgb / 255) * 1.)

        w, h = mask.shape[1], mask.shape[2]

        return get_box_mask(mask, loc, box_size, box_color, w, h, self.n_classes)


class domCueDataset(Dataset):
    """Dominoes with large box cues. Only supports CIFAR10 currently.
    Box pattern allowed to vary.
    """
    def __init__(self, 
                 dataset: CIFAR10, 
                 dataset_simple: Type[FashionMNIST], 
                 image_frac: float,
                 box_frac: float, 
                 mnist_frac: float,
                 randomize_img: bool = False,
                 randomize_box: bool = False,
                 randomize_mnist: bool = False,
                 box_cue_size: int = 4,
                 box_pattern: BoxPatternType = "MANDELBROT",
                 use_augmentation: bool = False,
                 augmentation_params: dict = None):
        self.dataset = dataset
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)
        self.box_cue_size = box_cue_size
        self.box_pattern = box_pattern
        self.use_augmentation = use_augmentation
        self.augmentation_params = augmentation_params

        self.image_frac, self.mnist_frac, self.box_frac =  image_frac, mnist_frac, box_frac
        self.randomize_img, self.randomize_box, self.randomize_mnist = randomize_img, randomize_box, randomize_mnist
        
        self.box_cue_ids: Dict[int, Dict[int, bool]] = get_cue_ids(labels=self.targets, n_classes=self.n_classes, cue_frac=self.box_frac)
        self.mnist_cue_ids = get_cue_ids(labels=self.targets, n_classes=self.n_classes, cue_frac=self.mnist_frac)

        self.association_ids: Dict[int, Dict[int, int]] = get_dominoes_associations(targets_c10=self.targets, targets_fmnist=np.array(dataset_simple.targets))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        image, label = self.dataset[item]
        image = self.create_domino(image, label, item)
        
        if self.use_augmentation:
            """Will return both labels."""
            image_2_idx = random.randint(0, self.__len__()-1)
            lam = random.betavariate(self.augmentation_params['alpha'], self.augmentation_params['beta'])
            
            image_2, label_2 = self.dataset[image_2_idx]
            image_2 = self.create_domino(image_2, label_2, image_2_idx)
            
            mixed_image = image.mul(lam).add(image_2, alpha=1-lam)
            
            return mixed_image, label, label_2, lam
        
        return image, label, None, None
    
    def create_domino(self, image: torch.Tensor, label: torch.Tensor, idx: int) -> torch.Tensor:
        """Label should be passed in as non-vector."""
        associated_id = self.association_ids[label][idx]
        if self.randomize_img:
            image = self.dataset[np.random.randint(0, len(self.dataset))][0]
        if self.image_frac < 1 and np.random.uniform() > self.image_frac:
            image = torch.zeros_like(image)

        if self.mnist_frac > 0:
            put_mnist = (np.random.uniform() < self.mnist_frac) if self.mnist_cue_ids is None else self.mnist_cue_ids[label][idx]
        else:
            put_mnist = False

        if self.box_frac > 0:
            put_box = (np.random.uniform() < self.box_frac) if self.box_cue_ids is None else self.box_cue_ids[label][idx]
        else:
            put_box = False
    
        if put_mnist:
            image_fmnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_mnist else self.dataset_simple[associated_id][0]
        else:
            image_fmnist = torch.zeros_like(image)

        if put_box:
            m = self.get_large_box(torch.zeros_like(image), label)

            # Zero image where mask is present and add mask
            image = m + (m == 0).all(axis=0) * image

        image = torch.cat((image_fmnist, image), dim=1)
        
        return image
    
    def get_large_box(self, mask: torch.Tensor, label: int) -> torch.Tensor:
        """Box is 1/3 of the width of the image.
        
        Args:
            mask: Default Torch tensor of zeros with same shape as image.
        
        Returns"
            mask: Torch tensor of zeros with box added.
        """
        loc = np.random.randint(0, 10) if self.randomize_box else (label % 10)
        box_size = mask.shape[1] // self.box_cue_size # Adjustable box cue size
        
        # Box patterns
        if self.box_pattern == BoxPatternType.MANDELBROT: # Pattern also depends on class, as well as position
            box_colors: List[np.ndarray] = generate_mandelbrot_images(num_images=self.n_classes, size=box_size)
            box_color = torch.from_numpy(box_colors[label % 10]).float()
        elif self.box_pattern == BoxPatternType.RANDOM:
            box_color = torch.rand((3, box_size, box_size))

        w, h = mask.shape[1], mask.shape[2] # Same dimensions as image

        return get_box_mask(mask, loc, box_size, box_color, w, h, self.n_classes)


def get_cue_ids(
    labels: np.ndarray = None,
    n_classes: int = 10,
    cue_frac: float = 1.0) -> Dict[int, Dict[int, bool]]:
    """
    Fix which samples of dataset will have cues.
        
    Returns:
        cue_ids: cue_ids[class_num][sample_id] = True if sample_id should have cue.
    """
    cue_ids = {}
    
    for class_num in range(n_classes):
        idx = np.where(labels == class_num)[0] # Indices in dataset where label matches class_num
        make_these_withcue = np.array([True]*idx.shape[0])
        make_these_withcue[int(idx.shape[0] * cue_frac):] = False # Make some of the samples not have cues
        cue_ids.update({class_num: {idx[sample_id]: make_these_withcue[sample_id] for sample_id in range(idx.shape[0])}})
        
    return cue_ids


def get_dominoes_associations(
    targets_fmnist: list[int],
    targets_c10: list[int]) -> Dict[int, Dict[int, int]]:
    """
    Make MNIST predictive for CIFAR10 by linking the indices of matching classes in each dataset.

    Args:
        targets_fmnist (numpy.ndarray): A 1-dimensional array of class labels for the Fashion MNIST dataset.
        targets_c10 (numpy.ndarray): A 1-dimensional array of class labels for the CIFAR10 dataset.

    Returns:
        dict: A dictionary where each key is a class number (0-9), and the value is a dictionary mapping indices
        in the CIFAR10 dataset to corresponding indices in the Fashion MNIST dataset where the class matches the key.
    """
    association_ids = {i: 0 for i in range(10)}
    
    for class_num in range(10):
        idx_c10 = np.where(targets_c10 == class_num)[0] # CIFAR10 labels match given class number
        idx_fmnist = np.where(targets_fmnist == class_num)[0] # MNIST labels match given class number
        association_ids[class_num] = {idx_c10[i]: idx_fmnist[i] for i in range(len(targets_c10) // 10)}
        
    return association_ids

        
class image3d(object):
    """Transforms for data."""
    def __call__(self, img):
        img = img.convert('RGB')
        return img

def get_transform(tform_type='nocue'):

    if(tform_type == 'nocue'):
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif(tform_type == 'dominoes'):
        train_transform = T.Compose([
            image3d(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ])

    return train_transform


def mandelbrot(c: complex, max_iter: int) -> int:
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter


def mandelbrot_set(
    xmin: float, 
    xmax: float, 
    ymin: float,
    ymax: float, 
    width: int,
    height: int,
    max_iter: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    return (r1,r2,np.array([[mandelbrot(complex(r, i),max_iter) for r in r1] for i in r2]))


def crop_image(image: np.ndarray, top_left: Tuple[int, int], size: int) -> np.ndarray:
    """Crop the image based on the top-left corner and desired size."""
    x, y = top_left
    return image[:, y:y+size, x:x+size]


def generate_mandelbrot_images(
    num_images: int,
    size: int,
    max_iter: int = 256) -> List[np.ndarray]:
    images = []
    for i in range(num_images):
        xmin = -2.0 + 0.2 * i
        xmax = 1.0 + 0.2 * i
        ymin = -1.5 + 0.2 * i
        ymax = 1.5 + 0.2 * i
        width = size
        height = size
        _, _, M = mandelbrot_set(xmin,xmax,ymin,ymax,width,height,max_iter)
        
        # Convert the grayscale values (iteration counts) to colors using a colormap.
        color_image = plt.cm.viridis(M / max_iter)[:, :, :3]  # RGB channels
        
        # Convert to format [channels x height x width]
        transposed_image = np.transpose(color_image, (2, 0, 1))
        
        images.append(transposed_image)
        
    return images


def get_box_dataloader(
    load_type: str = 'train',
    base_dataset: str = 'CIFAR10',
    cue_type: str = 'nocue',
    cue_frac: float = 1.0,
    randomize_cue: bool = False, 
    batch_size=64,
    data_dir: str = 'data',
    subset_ids: np.ndarray  = None,
    image_frac=1.0, box_frac=1.0, mnist_frac=1.0,
    randomize_img=False, randomize_box=False, randomize_mnist=False,
    box_cue_size: int = 4,
    use_augmentation: bool = False,
    augmentation_params: dict = None) -> DataLoader:
    """
    Return dataloaders for dominoes and box datasets. Main function to be called by other modules.
    
    Args:
        load_type: 'train' or 'test'
        base_dataset: 'CIFAR10', 'CIFAR100'
        cue_type: 'nocue', 'box', 'dominoes', 'domcues'
        box_frac, mnist_frac, image_frac, randomize_box, randomize_mnist: only for Dominoes
        
    Default download=False. Set download=True to download for first time.
    
    Doesn't matter what cue_type equals if base_dataset is 'Dominoes' - proportion of cues controlled by 'cue_fracs'.
    E.g. plain dominoes (CIFAR only predictive of label), use box_frac=0, mnist_frac=0.
    Shuffle automatically set to true for test and train.
    """
    if os.path.exists(f'{data_dir}'):
        download_datasets = False
    else:
        download_datasets = True
    is_train = (load_type=='train')
    
    # if base_dataset == 'Dominoes':
    #     base_dataset = 'CIFAR10'
    #     cue_frac = 0.0 if cue_type == 'nocue' else cue_frac
    #     cue_type = 'dominoes'
    ## 'Dominoes' base dataset automatically means cued dominoes for now
    if base_dataset == 'Dominoes':
        base_dataset = 'CIFAR10'
        cue_type = 'domcues'

    # Base dataset (pick train or test)
    dset_type = getattr(torchvision.datasets, base_dataset)
    dset = dset_type(root=f'{data_dir}/{base_dataset.lower()}/', 
                     train=(load_type =='train'), download=download_datasets, transform=get_transform('nocue'))

    # Cue type
    if (cue_type == 'nocue'):
        pass
    elif (cue_type == 'box'):
        dset = boxDataset(dset, cue_frac=cue_frac, randomize_cue=randomize_cue, randomize_img=randomize_img)
    elif (cue_type == 'dominoes'):
        dset_type: type[FashionMNIST] = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=is_train, download=download_datasets, transform=get_transform('dominoes'))
        dset = domDataset(
            dset,
            dset_simple,
            box_frac, 
            mnist_frac, 
            randomize_cue=randomize_cue, 
            randomize_img=randomize_img)
    elif (cue_type == 'domcues'):
        dset_type = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=is_train, download=download_datasets, transform=get_transform('dominoes'))
        dset = domCueDataset(
            dset, 
            dset_simple, 
            image_frac=image_frac, box_frac=box_frac, mnist_frac=mnist_frac,  randomize_img=randomize_img, randomize_box=randomize_box, randomize_mnist=randomize_mnist,
            box_cue_size=box_cue_size,
            use_augmentation=use_augmentation,
            augmentation_params=augmentation_params)

    if isinstance(subset_ids, np.ndarray):
        dset = torch.utils.data.Subset(dset, subset_ids)

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=is_train, drop_last=True)
    
    return dataloader


def show_images_grid(imgs_, class_labels, num_images):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(nrows, ncols, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            img = imgs_[ax_i]
            if img.ndim == 3 and img.shape[0] == 3:
                img = einops.rearrange(img, 'c h w -> h w c')
            ax.imshow(img, cmap='Greys_r', interpolation='nearest')
            
            # Get non-zero indices
            non_zero_indices = np.where(class_labels[ax_i])[0]
            label = ", ".join(map(str, non_zero_indices))
            ax.set_title(f'Non-zero indices: {label}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

        
if __name__ == "__main__":
    augmentation_params = {
    "alpha": 1.0,
    "beta": 1.0,
    "mix_prob": 0.9,
    "crop_prob": 0.5,
    "flip_prob": 0.5,
    "rotate_prob": 0.5
    }
    batch_size = 20
    train_loader = get_box_dataloader(load_type='test', base_dataset='Dominoes', batch_size=batch_size, cue_type='domcues', box_frac=1.0, mnist_frac=1.0, image_frac=0, randomize_box=False, randomize_mnist=True, randomize_img=False, box_cue_size=4, use_augmentation=True, augmentation_params=augmentation_params)
    for i, (x, y) in enumerate(train_loader):
        show_images_grid(x, y, num_images=batch_size)
        break
    
    
# Not currently in use
class domDataset(Dataset):
    """Dominoes CIFAR10 (complex) + FashionMNIST (simple) stacked vertically."""
    def __init__(self, 
                 dataset: Union[CIFAR10, CIFAR100],
                 dataset_simple: Type[FashionMNIST], 
                 image_frac: float = 1.0,
                 cue_frac: float = 1.0,
                 randomize_img=False,
                 randomize_cue: bool = False):
        self.dataset = dataset
        self.classes = dataset.classes
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        self.cue_frac, self.randomize_cue, self.randomize_img = cue_frac, randomize_cue, randomize_img
        self.cue_ids = get_cue_ids(labels=self.targets, n_classes=self.n_classes, cue_frac=cue_frac)

        self.association_ids: Dict = get_dominoes_associations(targets_c10=self.targets, targets_fmnist=np.array(dataset_simple.targets))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        associated_id = self.association_ids[label][item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        if self.cue_frac > 0:
            put_cue_attribute = (np.random.uniform() < self.cue_frac) if self.cue_ids is None else self.cue_ids[label][item]
        else:
            put_cue_attribute = False

        if put_cue_attribute:
            image_fmnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_cue else self.dataset_simple[associated_id][0]
        else:
            image_fmnist = torch.zeros_like(image)
        image = torch.cat((image_fmnist, image), dim=1)        

        return image, label