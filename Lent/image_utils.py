"""Custom datasets and augmentation methods for training and testing."""
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
import random
import os
from shapes_3D import *
import einops

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Dataset for inheritance for AugDataset

class AugShapes3D(Shapes3D):
    """Custom augmented dataset implementation. Inherits from other dataloaders, but must be multichannel images: [channels, height, width] OR [height, width, channels]
    Always check dimensions if unsure.
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
        """Turn x into torch tensor and permute to [c h w] format."""
        x, y = super().__getitem__(index)
        x = torch.from_numpy(x).contiguous()
        # Important - mixup needs to be implemented before other transforms
        x, y = self.mixup(x, y, self.alpha, self.beta)
        x = x.div(255.0)
        x = self.crop(x)
        x = self.flip(x)
        # x = self.rotate(x)
        if x.shape[0] != 3:
            x = einops.rearrange(x, 'h w c -> c h w')  # new shape: (3, 64, 64)
        print(y)
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
        x2 = torch.from_numpy(self.images[index,:,:,:]).contiguous()
        y2 = self.oh_labels[index, :]
        mixed_x = x.mul(lam).add(x2,alpha=1-lam)
        mixed_y = y.mul(lam).add(y2,alpha=1-lam)
        print(mixed_y)
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
        print("Angle", angle)
        x = TF.rotate(x, random.randint(0, angle))
        return x

if __name__ == "__main__":
    batch_size = 16
    aug_set = AugShapes3D(alpha=5, beta=3, mix_prob=0.5, crop_prob=0.5, crop_size=30, flip_prob=0.5, rotate_prob=0.5)
    aug_loader = DataLoader(aug_set, batch_size=batch_size, shuffle=True)
    iterator = iter(aug_loader)
    x, y = next(iterator)
    # Need to rearrange to [b h w c] for show_images_grid
    x = einops.rearrange(x, 'b c h w -> b h w c')
    show_images_grid(x, batch_size)

    ## Deprecated: batch mixup method. Superseded by mixup method that acts on individual images (ABOVE)
    # def mixup(self, x, y, alpha, beta):
    #     """Mixup function using beta distribution for mixing.
    #     If alpha = beta = 1, distribution uniform.
    #     If alpha > beta, distribution skewed towards 0.
    #     If alpha < beta, distribution skewed towards 1.
    #     When alpha > 1 and beta > 1, distribution is concentrated around 0 and 1 - more likely to interpolate samples with similar features.
    #     When alpha < 1 and beta < 1, distribution is concentrated around 0.5, = more likely to interpolate samples with dissimilar features.
    #     """
    #     if random.uniform(0,1) > self.mix_prob:
    #         return x, y
    #     batch_size = x.shape[0]
    #     # Lambda parameter, from beta distribution
    #     lam = torch.tensor([random.betavariate(alpha, beta) for _ in range(batch_size)])
    #     index = torch.randperm(batch_size)
    #     # Broadcast dimensions of lam to match x - which has shape [batch_size, channels, height, width]
    #     mixed_x = lam.view(batch_size, 1, 1, 1) * x + (1 - lam.view(batch_size, 1, 1, 1)) * x[index, :]
    #     mixed_y = lam.view(batch_size, 1) * y + (1 - lam.view(batch_size, 1)) * y[index, :]
    #     return mixed_x, mixed_y
    