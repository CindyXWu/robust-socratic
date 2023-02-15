"""Custom datasets and augmentation methods for training and testing."""
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import sys, os
import torch
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
output_dir = "data/32/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class CIFAR10Mixup(datasets.CIFAR10):
    def __init__(self, alpha, beta, *args, **kwargs):
        """Set parameters for beta distribution. 
        
        If alpha = beta = 1, distribution uniform.
        If alpha > beta, distribution skewed towards 0.
        If alpha < beta, distribution skewed towards 1.
        When alpha > 1 and beta > 1, distribution is concentrated around 0 and 1 - more likely to interpolate samples with similar features.
        When alpha < 1 and beta < 1, distribution is concentrated around 0.5, = more likely to interpolate samples with dissimilar features.
        """
        super(CIFAR10Mixup, self).__init__(*args, **kwargs)
        # Params for beta distribution
        self.alpha = alpha
        self.beta = beta

    def __getitem__(self, index):
        self.x, self.y = super().__getitem__(index)
        mixed_x, mixed_y = self.__mixup
        return mixed_x, mixed_y
       
    def __mixup(self):
        """Mixup functon for CIFAR_10."""
        batch_size = self.x.size()[0]
        # Lambda parameter, from beta distribution
        lam = torch.tensor([random.betavariate(self.alpha, self.beta) for _ in range(batch_size)])
        print(lam)
        index = torch.randperm(batch_size)
        # Broadcast dimensions of lam to match x - which has shape [batch_size, channels, height, width]
        mixed_x = lam.view(batch_size, 1, 1, 1) * self.x + (1 - lam.view(batch_size, 1, 1, 1)) * self.x[index, :]
        mixed_y = lam.view(batch_size, 1) * self.y + (1 - lam.view(batch_size, 1)) * self.y[index, :]
        return mixed_x, mixed_y
    
    def __crop(self):

if __name__ == '__main__':
    train_dataset = CIFAR10Mixup(root='./data', train=True, download=True, alpha=1.0, beta=1.0)
