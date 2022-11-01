import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Union, list, int
import pandas as pd
from error import FuncInputError

class vecDataset(Dataset):
    """Option to open dataset from existing CSV file or create from scratch."""
    def __init__(self, gen=False, filename=None, **kwargs):
        if not gen:
            try:
                if filename != None:
                    filename = kwargs.pop('filename')
                    self.dataset = np.genfromtxt(filename, delimiter=',')
                else:
                    raise FuncInputError
            except OSError:
                print("Error opening files")
        else:
            for key in ['simple', 'complex', 'numpoints']:
                if key not in kwargs:
                    raise FuncInputError

            num_points = kwargs.pop('num_points')
            self.simple = kwargs.pop('simple')
            complex = kwargs.pop('complex')

            # Generate data
            self.generate(self.simple, complex, num_points)
            
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
    
    def simple_randomise(self):
        for i in range(self.simple):
            np.random.shuffle(self.dataset[i,:])

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
                self.dataset[j][i] = np.random.choice([-1, 1])
            for j in range(len(complex)):
                self.dataset[simple+j][i] = self.n_slabs(complex[j], y)
            # TODO: add noise stuff

        name = input("Enter filename:")
        np.savetxt('Dataset {}'.format(name), self.dataset, delimiter=',')
    
    def n_slabs(n, y):
        """"Generate single x datapoint from single y label.
        
        Use k-point separation by dividing interval of length 2 into equally sized parts.

        :param n: number of slabs
        :param y: label
        """
        xs = np.linspace(-1, 1, n)
        if y == 1:
            x_poss = xs[::2]
            return np.random.choice(x_poss)
        elif y == -1:
            x_poss = xs[1::2]
            return np.random.choice(x_poss)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Note that csv file is stored as rows = features, cols = datapoints
        # To preserve x as col vectors
        input = self.dataset[:-1,idx]
        label = self.dataset[-1,idx]
        return input, label

def my_train_dataloader(gen=False, filename=None, simple=0, complex=0, num_points=0, mode=0, x=[]):
    """Used to load train and/or test splits.
    
    :param gen: Binary variable stating whether data should be generated or loading from an existing file. If so, filename should be given by filename parameter.
    :param mode: Gives mode for data randomisation.
        0: Generates without any randomisation.
        1: Randomises feature indices/index listed in x using split randomise. Randomises simple feature.
        2: Replaces simple component with uniform noise.
        3: Replaces simple component with uniform noise and split randomises complex components.
    :param x: Variable used for randomisation pass into split_randomise.

    :returns: Dataset of 1D input features and one single label y which is the LAST ROW of the dataset.
    """
    # Base: to test that SB is observed as comparison
    dset = vecDataset(gen, filename, simple, complex, num_points)
    # Enforce variance towards all complex features only
    if mode == 1:
        dset.split_randomise(x)
        dset.simple_randomise()
    # Turn simple into noise
    if mode == 2:
        for i in range(len(simple)):
            noise = np.random.uniform(-1, 1, num_points)
            dset[i,:] = noise
    if mode == 3:
        for i in range(len(simple)):
            noise = np.random.uniform(-1, 1, num_points)
            dset[i,:] = noise
        dset.split_randomise()

    return dset.dataset