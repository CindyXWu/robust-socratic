import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import einops

import torch.backends.cudnn as cudnn
torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


### With Cue Dataset Classes
# Box perturbations dataset
class boxDataset(Dataset):
    def __init__(self, dataset, cue_proportion=1., randomize_cue=False, randomize_img=False):

        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # cue information
        self.cue_proportion, self.randomize_cue, self.randomize_img = cue_proportion, randomize_cue, randomize_img
        self.cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=cue_proportion)

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        put_cue_attribute = (np.random.uniform() < self.cue_proportion) if self.cue_ids is None else self.cue_ids[label][item]
        
        if put_cue_attribute:
            m = self.get_box(torch.zeros_like(image), label)

            # Zero image where mask is present and add mask
            image = m + (m == 0).all(axis=0) * image

        return image, label
        
    # box creation function
    def get_box(self, mask, label):
        loc = np.random.randint(0, 10) if self.randomize_cue else (label % 10)

        l = (label // 10) if self.n_classes == 100 else 10   # only use color for CIFAR-100
        color = np.random.uniform() if (self.randomize_cue and self.n_classes == 100) else (l / 10) # only use color for CIFAR-100

        # HSV -> RGB -> BGR -> add w,h dimensions
        rgb = matplotlib.colors.hsv_to_rgb([color, 1, 255])[[2, 0, 1]][..., None, None]

        s = 3
        c = torch.Tensor((rgb / 255) * 1.)

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
    def __init__(self, dataset, dataset_simple, cue_proportion=1., randomize_cue=False, randomize_img=False):

        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # cue information
        self.cue_proportion, self.randomize_cue, self.randomize_img = cue_proportion, randomize_cue, randomize_img
        self.cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=cue_proportion)

        # association IDs
        self.association_ids = get_dominoes_associations(targets_c10=self.targets, targets_fmnist=np.array(dataset_simple.targets))

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        associated_id = self.association_ids[label][item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        if self.cue_proportion > 0:
            put_cue_attribute = (np.random.uniform() < self.cue_proportion) if self.cue_ids is None else self.cue_ids[label][item]
        else:
            put_cue_attribute = False

        if put_cue_attribute:
            image_fmnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_cue else self.dataset_simple[associated_id][0]
        else:
            image_fmnist = torch.zeros_like(image)
        image = torch.cat((image_fmnist, image), dim=1)        

        return image, label

# Dominoes dataset
class domCueDataset(Dataset):
    """Dataset for Dominoes with cues.
    Important: when passing in vars, box always comes first, then MNIST, then image.
    """
    def __init__(self, dataset, 
                 dataset_simple, 
                 box_frac, 
                 mnist_frac, 
                 image_frac=1.0, 
                 randomize_box=False,
                 randomize_mnist=False, 
                 randomize_img=False):
        self.dataset = dataset
        self.classes = dataset.classes
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # cue information
        self.mnist_frac, self.box_frac, self.image_frac, self.randomize_box, self.randomize_box, self.randomize_img = mnist_frac, box_frac, image_frac, randomize_box, randomize_mnist, randomize_img
        self.mnist_cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=self.mnist_frac)
        self.box_cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=self.box_frac)

        # association IDs
        self.association_ids = get_dominoes_associations(targets_c10=self.targets, targets_fmnist=np.array(dataset_simple.targets))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        associated_id = self.association_ids[label][item]

        if self.randomize_img:
            image = self.dataset[np.random.randint(0, len(self.dataset))][0]
        elif self.image_frac < 1 and np.random.uniform() > self.image_frac:
            # If image_frac < 1, then we randomly drop the image with probability 1 - image_frac
            image = torch.zeros_like(image)
        else:
            image = image

        if self.mnist_frac > 0:
            put_mnist = (np.random.uniform() < self.mnist_frac) if self.mnist_cue_ids is None else self.mnist_cue_ids[label][item]
        else:
            put_mnist = False

        if self.box_frac > 0:
            put_box = (np.random.uniform() < self.box_frac) if self.box_cue_ids is None else self.box_cue_ids[label][item]
        else:
            put_box = False
    
        if put_mnist:
            image_fmnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_mnist else self.dataset_simple[associated_id][0]
        else:
            image_fmnist = torch.zeros_like(image)     

        if put_box:
            m = self.get_box(torch.zeros_like(image), label)

            # Zero image where mask is present and add mask
            image = m + (m == 0).all(axis=0) * image

        image = torch.cat((image_fmnist, image), dim=1)   
        
        return image, label
    
    # box creation function
    def get_box(self, mask, label):
        loc = np.random.randint(0, 10) if self.randomize_box else (label % 10)

        l = (label // 10) if self.n_classes == 100 else 10   # only use color for CIFAR-100
        color = np.random.uniform() if (self.randomize_box and self.n_classes == 100) else (l / 10) # only use color for CIFAR-100

        # HSV -> RGB -> BGR -> add w,h dimensions
        rgb = matplotlib.colors.hsv_to_rgb([color, 1, 255])[[2, 0, 1]][..., None, None]

        s = 3
        c = torch.Tensor((rgb / 255) * 1.)

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


#### Fix which samples of dataset will have cues
def get_cue_ids(targets=None, n_classes=10, prob=1.):
    """Returns:
        cue_ids: Dictionary of dictionaries. cue_ids[class_num][sample_id] = True if sample_id should have cue.
    """
    cue_ids = {}
    for class_num in range(n_classes):
        idx = np.where(targets == class_num)[0] # Indices in dataset where label matches class_num
        make_these_withcue = np.array([True]*idx.shape[0])
        make_these_withcue[int(idx.shape[0] * prob):] = False # Make some of the samples not have cues
        cue_ids.update({class_num: {idx[sample_id]: make_these_withcue[sample_id] for sample_id in range(idx.shape[0])}})
    return cue_ids


#### Dominoes data dictionaries
def get_dominoes_associations(targets_fmnist, targets_c10):
    """Match the indices of the two datasets to create associations between the two datasets.
    This makes MNIST predictive for CIFAR10.
    """
    association_ids = {i: 0 for i in range(10)}
    for class_num in range(10):
        idx_c10 = np.where(targets_c10 == class_num)[0] # cifar10 labels match given class number
        idx_fmnist = np.where(targets_fmnist == class_num)[0] # mnist labels match given class number
        association_ids[class_num] = {idx_c10[i]: idx_fmnist[i] for i in range(len(targets_c10) // 10)}
    return association_ids

        
### Transforms for data 
class image3d(object):
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


### Dataloaders
def get_box_dataloader(load_type='train', base_dataset='CIFAR10', cue_type='nocue', cue_proportion=1.0, randomize_cue=False, 
                    randomize_img=False, batch_size=64, data_dir='data', subset_ids=None, box_frac=1.0, mnist_frac=1.0, image_frac=1.0, randomize_box=False, randomize_mnist=False):
    """Args:
        load_type: 'train' or 'test'
        base_dataset: 'CIFAR10', 'CIFAR100'
        cue_type: 'nocue', 'box', 'dominoes', 'domcues'
        box_frac: only used for domcues
        mnist_frac: only used for domcues
        randomize_cues: used only for domcues
    Datasets default download=False. Set download=True to download for first time.
    It does not matter what you set cue_type to if base_dataset is 'Dominoes' - the proportion of cues/presence is controlled by 'cue_proportions'.
    E.g. if you want plain dominoes (CIFAR only predictive of label), use box_frac=0, mnist_frac=0.
    Shuffle automatically set to true for test and train.
    """
    ## 'Dominoes' base dataset automatically means cued dominoes for now
    # if base_dataset == 'Dominoes':
    #     base_dataset = 'CIFAR10'
    #     cue_proportion = 0.0 if cue_type == 'nocue' else cue_proportion
    #     cue_type = 'dominoes'

    if base_dataset == 'Dominoes':
        base_dataset = 'CIFAR10'
        cue_type = 'domcues'

    # define base dataset (pick train or test)
    dset_type = getattr(torchvision.datasets, base_dataset)
    dset = dset_type(root=f'{data_dir}/{base_dataset.lower()}/', 
                     train=(load_type =='train'), download=False, transform=get_transform('nocue'))

    # pick cue
    if (cue_type == 'nocue'):
        pass
    elif (cue_type == 'box'):
        dset = boxDataset(dset, cue_proportion=cue_proportion, randomize_cue=randomize_cue, randomize_img=randomize_img)
    elif (cue_type == 'dominoes'):
        dset_type = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=(load_type == 'train'), download=False, transform=get_transform('dominoes'))
        dset = domDataset(dset, dset_simple, box_frac, mnist_frac, randomize_cue=randomize_cue, randomize_img=randomize_img)
    elif (cue_type == 'domcues'):
        dset_type = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=(load_type == 'train'), download=False, transform=get_transform('dominoes'))
        dset = domCueDataset(dset, dset_simple, box_frac=box_frac, mnist_frac=mnist_frac, image_frac=image_frac, randomize_box=randomize_box, randomize_mnist=randomize_mnist, randomize_img=randomize_img)

    if isinstance(subset_ids, np.ndarray):
        dset = torch.utils.data.Subset(dset, subset_ids)

    # define dataloader
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader


def show_images_grid(imgs_, class_labels, num_images=25):
    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs_[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_title(f'Class: {class_labels[ax_i]}')  # Display the class label as title
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')
    plt.show()


class LR_Scheduler(object):
    def __init__(self, optimizer, num_epochs, base_lr, final_lr, iter_per_epoch):
        self.base_lr = base_lr
        # Iterations per epoch
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
    
if __name__ == "__main__":
    train_loader = get_box_dataloader(load_type='train', base_dataset='Dominoes Box', batch_size=64, cue_type='domcues', cue_proportion=0.5, randomize_cue=True, cue_proportions=[1,1], randomize_cues=[False, False])
    for i, (x, y) in enumerate(train_loader):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        show_images_grid(x, y, num_images=64)
        break
