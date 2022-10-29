import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
import numpy as np
import pickle as pkl
from typing import Union, list, int
import pandas as pd
import os, sys
from error import FuncInputError

import torch.backends.cudnn as cudnn
torch.manual_seed(0)    # Set seed for generating random numbers, returning torch.Generator object
cudnn.deterministic = True
cudnn.benchmark = False

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

class vecDataset(Dataset):
    def __init__(self, gen=False, **kwargs):
        if not gen:
            try:
                if 'filename' in kwargs:
                    filename = kwargs.pop('filename')
                self.dataset = np.genfromtxt(filename, delimiter=',')
            except OSError:
                print("Error opening files")
        else:
            for key in ['simple', 'complex', 'noise', 'numpoints']:
                if key not in kwargs:
                    raise FuncInputError

            num_points = kwargs.pop('num_points')
            simple = kwargs.pop('simple')
            complex = kwargs.pop('complex')
            noise = kwargs.pop('noise')

            self.generate(simple, complex, noise, num_points)
            

    def split_randomise(self, x):
        """Randomise n-fractions of the dataset.
        
        Each subset is sensitive to one predictive feature.

        :param x: list of indices to be correlated with in each subset
        """
        n = len(x)  # Num of chunks
        chunk_size = self.__len__()//n  # Size of each chunk
        for i in range(len(x)):
            idx = i*chunk_size
            np.random.shuffle(self.dataset[x[i],idx:idx+chunk_size])

    def generate(self, simple, complex, noise, N):
        """"Generates dataset if it isn't read from file.
        
        :param complex: list of number of slabs to use for each feature
        """
        # Rows: features
        # Cols: datapoints
        self.dataset = np.empty(N, simple+complex+noise+1)
        for i in range(N):
            y = np.random.choice[-1, 1]
            # Last row is y
            self.dataset[-1][i] = y
            for j in range(simple):
                self.dataset[j][i] = np.random.choice([])
            for j in range(len(complex)):
                self.dataset[simple+j][i] = self.n_slabs(complex[j], y)
            # TODO: add noise stuff
    
    def n_slabs(n, y):
        """"Generate single x datapoint from single y label.
        
        Use k-point separation by dividing interval of length 2 into equally sized parts.

        :param n: number of slabs
        :param y: label
        """
        xs = np.linspace(0, 2, n)
        if y == 1:
            x_poss = xs[::2]-1
            return np.random.choice(x_poss)
        elif y == -1:
            x_poss = xs[1::2]-1
            return np.random.choice(x_poss)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Note that csv file is stored as rows = features, cols = datapoints
        # To preserve x as col vectors
        input = self.dataset[:,idx][:-1]
        label = self.dataset[:,idx][-1]
        return input, label

vecDataset(gen=True, simple=1, complex=[3,5], noise=0, numpoints=100)
