"""Custom datasets and augmentation methods for training and testing.
AugMix implemented from Hendrycks et al. (2019) https://arxiv.org/abs/1912.02781"""
import torchvision.datasets as datasets
import torchvision.transforms.functional as TF
import torch
import random

class CIFAR10Aug(datasets.CIFAR10):
    """Custom augmented CIFAR10 implementation."""
    def __init__(self, alpha, beta, mix_prob, crop_prob, flip_prob, rotate_prob, jitter_prob, erase_prob, *args, **kwargs):
        """Set parameters for transform probability."""
        super(CIFAR10Aug, self).__init__(*args, **kwargs)
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
        x, y = super().__getitem__(index)
        x, y = mixup(x, y, self.alpha, self.beta, self.mix_prob)
        x = crop(x, self.crop_prob, x.size()[1])
        x = flip(x, self.flip_prob)
        x = rotate(x, self.rotate_prob)
        x = jitter(x, self.jitter_prob)
        x = TF.randomerasing(x, p=self.erase_prob, scale=(0.02, 0.1), ratio=(0.5, 2), value=0)
        return x, y
        
def mixup(x, y, alpha, beta, mix_prob):
    """Mixup function using beta distribution for mixing.
    If alpha = beta = 1, distribution uniform.
    If alpha > beta, distribution skewed towards 0.
    If alpha < beta, distribution skewed towards 1.
    When alpha > 1 and beta > 1, distribution is concentrated around 0 and 1 - more likely to interpolate samples with similar features.
    When alpha < 1 and beta < 1, distribution is concentrated around 0.5, = more likely to interpolate samples with dissimilar features.
    """
    if random.uniform(0,1) > mix_prob:
        return x, y
    batch_size = x.size()[0]
    # Lambda parameter, from beta distribution
    lam = torch.tensor([random.betavariate(alpha, beta) for _ in range(batch_size)])
    index = torch.randperm(batch_size)
    # Broadcast dimensions of lam to match x - which has shape [batch_size, channels, height, width]
    mixed_x = lam.view(batch_size, 1, 1, 1) * x + (1 - lam.view(batch_size, 1, 1, 1)) * x[index, :]
    mixed_y = lam.view(batch_size, 1) * y + (1 - lam.view(batch_size, 1)) * y[index, :]
    return mixed_x, mixed_y
    
def crop(x, crop_prob, crop_size):
    """Randomly crop images and return resized to 32x32
    Args:
        crop_prob: float, probability of applying crop
        crop_size: int, size of square crop in pixels
    """
    if random.uniform(0,1) > crop_prob:
        return x
    _, h, w = x.size()
    # Randomly select top left corner of crop's y coordinate
    i = random.randint(0, w - crop_size)
    # Randomly select top left corner of crop's x coordinate
    j = random.randint(0, h - crop_size)
    cropped = TF.crop(x, j, i, crop_size, crop_size)
    return TF.resize(cropped, (32, 32))

def flip(x, flip_prob):
    """Randomly flip images both horizontally and vertically with probability flip_prob.
    """
    if random.uniform(0,1) < flip_prob:
        x = TF.hflip(x)
    if random.uniform(0,1) < flip_prob:
        x = TF.vflip(x)
    return x

def rotate(x, rotate_prob):
    """Randomly rotate images with probability rotate_prob by a random angle."""
    if random.uniform(0,1) > rotate_prob:
        return x
    angle = random.randint(0, 360)
    return TF.rotate(x, random.randint(0, angle))

def jitter(x, jitter_prob):
    """Randomly colour jitter images with probability jitter_prob."""
    if random.uniform(0,1) > jitter_prob:
        return x
    brightness = [0.3, 0.5]
    saturation = [0.3, 0.5]
    contrast = [0.4, 0.5]
    hue = [0.1, 0.3]
    return TF.colorjitter(x, brightness, contrast, saturation, hue)

if __name__ == '__main__':
    train_dataset = CIFAR10Aug(alpha=1.0, beta=1.0, mix_prob=0.3, crop_prob=0.3, flip_prob=0.3, rotate_prob=0.3, jitter_prob=0.3, erase_prob=0.3, root='./data/aug', download=True, train=True)