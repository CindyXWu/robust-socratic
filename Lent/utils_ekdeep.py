import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
import numpy as np
import pickle as pkl

import torch.backends.cudnn as cudnn
torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


### Spurious dataset classes
# Color perturbations dataset
class coloredDataset(Dataset):
    def __init__(self, dataset, spurious_corr=1., randomize_color=False, use_spurious_by_ids=False, reverse_color=False, randomize_img=False):
        """Args:
            dataset: 
            use_spurious_by_ids: 'rand' or 'norand' or False (for now set to False) 
        """
        self.dataset = dataset
        self.classes = dataset.classes
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)
        self.spurious_corr = spurious_corr
        self.randomize_color = randomize_color
        self.randomize_img = randomize_img
        if use_spurious_by_ids == 'rand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=True, prob=spurious_corr)
        elif use_spurious_by_ids == 'norand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=False, prob=spurious_corr)
        else:
            self.spurious_ids = None

        self.colors = np.array(
                                    [[1., 1., 1.], 
                                    [0.12156863, 0.46666667, 0.70588235], 
                                    [1.        , 0.49803922, 0.05490196],
                                    [0.17254902, 0.62745098, 0.17254902],
                                    [0.83921569, 0.15294118, 0.15686275],
                                    [0.58039216, 0.40392157, 0.74117647],
                                    [0.54901961, 0.3372549,  0.29411765],
                                    [0.89019608, 0.46666667, 0.76078431],
                                    [0.49803922, 0.49803922, 0.49803922],
                                    [0.7372549,  0.74117647, 0.13333333],
                                    [0.09019608, 0.74509804, 0.81176471],]
                                )
        if reverse_color:
            self.colors = np.array(
                                        [[1., 1., 1.], 
                                        [0.81176471, 0.74509804, 0.09019608],
                                        [ 0.13333333, 0.74117647, 0.7372549],
                                        [0.49803922, 0.49803922, 0.49803922],
                                        [0.76078431, 0.46666667, 0.89019608],
                                        [0.29411765, 0.3372549,  0.54901961],
                                        [0.74117647, 0.40392157, 0.58039216],
                                        [0.15686275, 0.15294118, 0.83921569],
                                        [0.17254902, 0.62745098, 0.17254902],
                                        [0.05490196, 0.49803922, 1.],
                                        [0.70588235, 0.46666667, 0.12156863], ]
                                    )


    # get spurious sample IDs
    def get_spurious_ids(self):
        with open(self.spurious_path, 'rb') as f:
            spurious_ids = pkl.load(f)
        return spurious_ids

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image
        l = int(10 * np.random.uniform()) if self.randomize_color else label
        c = torch.Tensor(self.colors[l+1]).reshape(-1, 1, 1)

        put_spurious_feature = np.random.uniform() < self.spurious_corr if self.spurious_ids is None else self.spurious_ids[label][item]
        if(put_spurious_feature):
            mask = (image > 1/255)
            image *= mask * c
        return image, label



# Box perturbations dataset
class boxDataset(Dataset):
    def __init__(self, dataset, spurious_corr=1., spurious_intensity=1., spurious_boxsize=3, randomize_intensity=False, 
                randomize_size=False, randomize_loc=False, randomize_color=False, randomize_bgd=False, use_spurious_by_ids=False):
        """Args:
            spurious_corr: probability of box spurious feature
        """
        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # spuriousness attributes
        self.spurious_intensity = spurious_intensity
        self.spurious_corr = spurious_corr
        self.spurious_boxsize = spurious_boxsize
        self.randomize_intensity = randomize_intensity
        self.randomize_size = randomize_size
        self.randomize_loc = randomize_loc
        self.randomize_color = randomize_color
        self.randomize_bgd = randomize_bgd
        if use_spurious_by_ids == 'rand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=True, prob=spurious_corr)
        elif use_spurious_by_ids == 'norand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=False, prob=spurious_corr)
        else:
            self.spurious_ids = None

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        # If randomise background then pick random image
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_bgd else image

        put_spurious_feature = (np.random.uniform() < self.spurious_corr) if self.spurious_ids is None else self.spurious_ids[label][item]
        
        if put_spurious_feature:
            m = self.get_box(torch.zeros_like(image), label)

            # Zero image where mask is present and add mask
            image = m + (m == 0).all(axis=0) * image

        return image, label
        
    # box creation function
    def get_box(self, mask, label):
        # If randomise location else location predictive of label
        loc = np.random.randint(0, 10) if self.randomize_loc else (label % 10)

        l = (label // 10) if self.n_classes == 100 else 10   # only use color for CIFAR-100
        color = np.random.uniform() if self.randomize_color else (l / 10)

        # HSV -> RGB -> BGR -> add w,h dimensions
        rgb = matplotlib.colors.hsv_to_rgb([color, 1, 255])[[2, 0, 1]][..., None, None]

        s = np.random.randint(0, self.spurious_boxsize + 5) if self.randomize_size else self.spurious_boxsize
        intensity = np.random.uniform() if self.randomize_intensity else self.spurious_intensity
        c = torch.Tensor((rgb / 255) * intensity)

        w, h = mask.shape[1], mask.shape[2]

        if (loc == 0): # upper left
            mask[:, :s, :s] = c
        elif (loc == 1): # upper center
            mask[:, :s, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 2): # upper right
            mask[:, :s, -s:] = c
            
        elif (loc == 3): # middle left
            mask[:, (w - s) // 2 : (w + s) // 2, :s] = c
        elif (loc == 4): # middle center
            mask[:, (w - s) // 2 : (w + s) // 2, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 5): # middle right
            mask[:, (w - s) // 2 : (w + s) // 2, -s:] = c
            
        elif (loc == 6): # lower left
            mask[:, -s:, :s] = c
        elif (loc == 7): # lower center
            mask[:, -s:, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 8): # lower right
            mask[:, -s:, -s:] = c
            
        elif (loc == 9):
            if self.n_classes == 10:
                # draw nothing if CIFAR-10
                pass
            else:
                # mid-upper-left if CIFAR-100
                mask[:, (w - s) // 2 - (w // 4) : 
                        (w + s) // 2 - (w // 4), 
                        (h - s) // 2 - (h // 4) : 
                        (h + s) // 2 - (h // 4)] = c

        return mask


# Dominoes dataset
class domDataset(Dataset):
    def __init__(self, dataset, dataset_simple, spurious_corr=1., randomize_bgd=False, use_spurious_by_ids=False, randomize_img=False):

        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # spuriousness attributes
        self.spurious_corr = spurious_corr
        self.randomize_cue = randomize_bgd
        self.randomize_img = randomize_img
        if use_spurious_by_ids == 'rand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=True, prob=spurious_corr)
        elif use_spurious_by_ids == 'norand':
            self.spurious_ids = get_spurious_ids(targets=self.targets, n_classes=self.n_classes, use_rand=False, prob=spurious_corr)
        else:
            self.spurious_ids = None

        # association IDs
        self.association_ids = get_dominoes_associations(targets_c10=self.targets, targets_mnist=np.array(dataset_simple.targets))

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        associated_id = self.association_ids[label][item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        if self.spurious_corr > 0:
            put_spurious_feature = (np.random.uniform() < self.spurious_corr) if self.spurious_ids is None else self.spurious_ids[label][item]
        else:
            put_spurious_feature = False

        if put_spurious_feature:
            image_mnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_cue else self.dataset_simple[associated_id][0]
        else:
            image_mnist = torch.zeros_like(image)
        image = torch.cat((image_mnist, image), dim=1)        

        return image, label


        
### Transforms for data 
# Expansion transform needed for MNIST
class mnist3d(object):
    def __call__(self, img):
        img = img.convert('RGB')
        return img

def get_transform(tform_type='plain'):
    """Args:
        tform_type: 'plain' or 'MNIST' or 'dominoes' depending on dataset used."""
    if(tform_type == 'plain'):
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif(tform_type == 'MNIST'):
        train_transform = T.Compose([
            mnist3d(),
            T.ToTensor(),
        ])

    elif(tform_type == 'dominoes'):
        train_transform = T.Compose([
            mnist3d(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ])

    return train_transform


### Dataloaders
def get_dataloader(load_type='train', base_dataset='CIFAR10', spurious_type='plain', spurious_corr=1.0, spurious_intensity=1., 
                    spurious_boxsize=3, tform_type='plain', randomize_intensity=False, randomize_size=False, randomize_loc=False, 
                    randomize_bgd=False, randomize_color=False, batch_size=64, use_spurious_by_ids=False, randomize_img=False,
                    data_dir='data', finding_connectivity=False, subset_ids=None):
    """Returns dataloader for specified dataset.
    Args:
        load_type: 'train' or 'test'
        base_dataset: e.g. 'CIFAR10', 'CIFAR100', 'MNIST'
        spurious_type: 'plain', 'color', 'reverse_color', 'box', 'dominoes'
        spurious_corr: probability of spurious feature
        spurious_intensity: can set or randomise intensity of each pixel in box
        spurious_boxsize: as on box, size of box...
        tform_type: transform type for get_transform()
        randomise_loc: whether to randomise box location (if set False, then box location is label % 10; 10 possibilities represent nothing/9 locations in grid for CIFAR-10)
        randomize_bgd: if True then pick random image not predictive of label
        randomize_color: if True then pick random color for box
        use_spurious_by_ids: 
    """
    # define transforms
    is_train = (load_type == 'train')
    transform = get_transform(tform_type if base_dataset != 'MNIST' else 'MNIST')

    if base_dataset == 'Dominoes':
        base_dataset = 'CIFAR10'
        spurious_corr = 0.0 if spurious_type == 'plain' else spurious_corr
        spurious_type = 'dominoes'

    # define base dataset (pick train or test)
    dset_type = getattr(torchvision.datasets, base_dataset)
    dset = dset_type(root=f'{data_dir}/{base_dataset.lower()}/', 
                     train=is_train, download=True, transform=transform)

    # pick normal vs. spurious
    if (spurious_type == 'plain'):
        pass
    elif (spurious_type == 'color' or spurious_type == 'reverse_color'): # intensity is fixed to 0. in normal operation mode
        dset = coloredDataset(dset, spurious_corr=spurious_corr, randomize_color=randomize_color, use_spurious_by_ids=use_spurious_by_ids, 
                                reverse_color=(spurious_type=='reverse_color'), randomize_img=randomize_img)
    elif (spurious_type == 'box'):
        dset = boxDataset(dset, spurious_corr=spurious_corr, spurious_intensity=spurious_intensity, spurious_boxsize=spurious_boxsize,
                          randomize_intensity=randomize_intensity, randomize_loc=randomize_loc, randomize_bgd=randomize_bgd, 
                          randomize_size=randomize_size, randomize_color=randomize_color, use_spurious_by_ids=use_spurious_by_ids)
    elif (spurious_type == 'dominoes'):
        dset_type = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=is_train, download=True, transform=get_transform('dominoes'))

        dset = domDataset(dset, dset_simple, spurious_corr=spurious_corr, randomize_bgd=randomize_bgd, randomize_img=randomize_img, use_spurious_by_ids=use_spurious_by_ids)

    if isinstance(subset_ids, np.ndarray):
        dset = torch.utils.data.Subset(dset, subset_ids)

    # define dataloader
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=(load_type=='train'), num_workers=2, drop_last=True)
    return dataloader


#### Change setup
def change_setup(base_setup, attr_change=[], attr_val=[]):
    setup = base_setup.copy()
    for attr, val in zip(attr_change, attr_val):
        setup[attr] = val
    return setup


#### Create spurious dataset by IDs
def get_spurious_ids(targets=None, n_classes=10, use_rand=False, prob=1.):
    spurious_ids = {}
    for class_num in range(n_classes):
        # Indexes where label is of target class
        idx = np.where(targets == class_num)[0]
        # Masked True array
        make_these_spurious = np.array([True]*idx.shape[0])
        if(use_rand):
            # Bernoulli coin toss to set some of these True values False with probability (1-p)
            make_these_spurious *= (np.random.binomial(n=1, p=prob, size=make_these_spurious.shape) > 0)
        else:
            # First p fraction are False
            make_these_spurious[int(idx.shape[0] * prob):] = False

        # Dictionary of dictionaries
        spurious_ids.update({class_num: {idx[sample_id]: make_these_spurious[sample_id] for sample_id in range(idx.shape[0])}})

    return spurious_ids


#### Dominoes data dictionaries
def get_dominoes_associations(targets_mnist, targets_c10):
    association_ids = {i: 0 for i in range(10)}
    for class_num in range(10):
        idx_c10 = np.where(targets_c10 == class_num)[0]
        idx_mnist = np.where(targets_mnist == class_num)[0]
        association_ids[class_num] = {idx_c10[i]: idx_mnist[i] for i in range(len(targets_c10) // 10)}
    return association_ids


class LR_Scheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        decay_iter = iter_per_epoch * num_epochs
        self.lr_schedule = final_lr+0.5*(base_lr-final_lr)*(1+np.cos(np.pi*np.arange(decay_iter)/decay_iter))        
        self.optimizer = optimizer
        self.iter = 0
        self.current_lr = 0

    def step(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        self.current_lr = lr

    def get_lr(self):
        return self.current_lr