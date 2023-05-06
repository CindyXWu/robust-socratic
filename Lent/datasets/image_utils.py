"""Custom datasets and augmentation methods for training and testing."""
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
import torchvision.datasets as datasets
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import random
import os
from shapes_3D import *
import einops

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Dataset for inheritance for AugDataset

class AugShapes3D(Shapes3D):
    """Custom augmented dataset implementation for 3D shapes dataset.
    Includes transforms from numpy to tensor.
    Also includes handling of dimensions (automatically [h w c ] for base class, but PyTorch prefers [c h w]).
    Returns:
        x: torch tensor
        y: soft label tensor of size num_classes
    """
    def __init__(self, alpha, beta, mix_prob, crop_prob, crop_size, flip_prob, rotate_prob):
        super().__init__()
        """Parent class needs a self.images and self.labels attribute."""
        # Params for beta distribution
        self.alpha = alpha
        self.beta = beta
        self.mix_prob = mix_prob
        self.crop_prob = crop_prob
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        Returns:
            x: c w h tensor
            y: 8*1 tensor of soft labels
        """
        x, y = super().__getitem__(index)
        y = F.one_hot(torch.tensor(y).to(torch.int64), num_classes=self.num_classes)
        # Important - mixup needs to be implemented before other transforms
        x, y = self.mixup(x, y, self.alpha, self.beta)
        x = self.crop(x)
        x = self.flip(x)
        # x = self.rotate(x)
        if x.shape[0] != 3:
            x = einops.rearrange(x, 'h w c -> c h w')  # Ensure shape: (3, 64, 64)
        return x, y

    def __len__(self):
        return super().__len__()
            
    def mixup(self, x, y, alpha, beta):
        """Mixup function using beta distribution for mixing.
        If alpha = beta = 1, distribution uniform.
        If alpha >> beta, lam distribution skewed towards 1.
        If alpha << beta, lam distribution skewed towards 0.
        """
        if torch.rand(1) > self.mix_prob:
            return x, y
        index = random.randint(0, self.__len__()-1)
        # Lambda parameter, from beta distribution
        lam = random.betavariate(alpha, beta)
        # Get random sample from dataset
        x2, y2 = self.__getitem__(index)
        y2 = F.one_hot(torch.tensor(y2).to(torch.int64), num_classes=self.num_classes)
        mixed_x = x.mul(lam).add(x2,alpha=1-lam)
        mixed_y = y.mul(lam).add(y2,alpha=1-lam)
        return mixed_x, mixed_y
    
    def crop(self, x):
        """Randomly crop images and return resized to original size."""
        if random.uniform(0,1) > self.crop_prob:
            return x
        if x.shape[0] != 3:
            x = einops.rearrange(x, 'h w c -> c h w')  # new shape: (3, 64, 64)
        h = w = x.shape[1]
        # Randomly select top left corner of crop's y coordinate
        i = random.randint(0, w - self.crop_size)
        # Randomly select top left corner of crop's x coordinate
        j = random.randint(0, h - self.crop_size)
        cropped = TF.crop(x, j, i, self.crop_size, self.crop_size)
        resized = TF.resize(cropped, (w, h))
        assert resized.shape == (3, w, h)
        return resized

    def flip(self, x):
        """Randomly flip images both horizontally and vertically with probability flip_prob."""
        if random.uniform(0,1) < self.flip_prob:
            x = TF.hflip(x)
        if random.uniform(0,1) < self.flip_prob:
            x = TF.vflip(x)
        return x

    def rotate(self, x):
        """Randomly rotate images with probability rotate_prob by a random angle."""
        if random.uniform(0,1) > self.rotate_prob:
            return x
        angle = random.randint(0, 45)
        x = TF.rotate(x, random.randint(0, angle))
        return x

class AugCIFAR(datasets.CIFAR10):
        """Custom augmented CIFAR10 implementation."""
        def __init__(self, alpha, beta, mix_prob, crop_prob, flip_prob, rotate_prob, jitter_prob, erase_prob, *args, **kwargs):
            """Set parameters for transform probability."""
            super(AugCIFAR, self).__init__(*args, **kwargs)
            # Params for beta distribution
            self.alpha = alpha
            self.beta = beta
            self.mix_prob = mix_prob
            self.crop_prob = crop_prob
            self.flip_prob = flip_prob
            self.rotate_prob = rotate_prob
            self.jitter_prob = jitter_prob
            self.erase_prob = erase_prob

        def __getitem__(self, index):
            """Turn x into torch tensor and permute to [c h w] format."""
            x, y = super().__getitem__(index)
            y = F.one_hot(torch.tensor(y), 10)
            x, y = self.mixup(x, y, self.alpha, self.beta)
            x = self.crop(x)
            x = self.flip(x)
            # x = self.rotate(x)
            print(x, y)
            return x, y

        def mixup(self, x, y, alpha, beta):
            if torch.rand(1) > self.mix_prob:
                return x, y
            index = random.randint(0, self.__len__()-1)
            # Lambda parameter, from beta distribution
            lam = random.betavariate(alpha, beta)
            # Return PIL.Image.Image, int
            x2, y2 = self.__getitem__(index)
            mixed_x = x.mul(lam).add(x2,alpha=1-lam)
            mixed_y = y.mul(lam).add(F.one_hot(torch.tensor(y2), 10),alpha=1-lam)
            return mixed_x, mixed_y

        def crop(self, x):
            """Randomly crop images and return resized to original size."""
            if random.uniform(0,1) > self.crop_prob:
                return x
            h = w = x.shape[1]
            # Randomly select top left corner of crop's y coordinate
            i = random.randint(0, w - self.crop_size)
            # Randomly select top left corner of crop's x coordinate
            j = random.randint(0, h - self.crop_size)
            cropped = TF.crop(x, j, i, self.crop_size, self.crop_size)
            return TF.resize(cropped, (w, h))

        def flip(self, x):
            """Randomly flip images both horizontally and vertically with probability flip_prob."""
            if random.uniform(0,1) < self.flip_prob:
                x = TF.hflip(x)
            if random.uniform(0,1) < self.flip_prob:
                x = TF.vflip(x)
            return x

        def rotate(self, x):
            """Randomly rotate images with probability rotate_prob by a random angle."""
            if random.uniform(0,1) > self.rotate_prob:
                return x
            angle = random.randint(0, 45)
            x = TF.rotate(x, random.randint(0, angle))
            return x
        
def load_cifar_10(dims):
    """Load CIFAR-10 dataset and return dataloaders.
    :param dims: tuple, dimensions of the images
    """
    transform = transforms.Compose([transforms.Resize((dims[0],dims[1])),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,  
                                0.406], [0.229, 0.224, 0.225])])
    trainset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=True, transform=transform)
    testset = datasets.CIFAR10(root='./data/'+str(dims[0]), download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
    return trainset, testset, trainloader, testloader

if __name__ == "__main__":
    batch_size = 16
    aug_set = AugShapes3D(alpha=5, beta=3, mix_prob=0.5, crop_prob=0.5, crop_size=30, flip_prob=0.5, rotate_prob=0.5)
    aug_loader = DataLoader(aug_set, batch_size=batch_size, shuffle=True)
    iterator = iter(aug_loader)
    x, y = next(iterator)
    # Need to rearrange to [b h w c] for show_images_grid
    x = einops.rearrange(x, 'b c h w -> b h w c')
    show_images_grid(x, batch_size)
