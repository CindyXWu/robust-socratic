import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from error import *
import sys, os
from plotting import *

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
output_dir = "data/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"

class vecDataset(Dataset):
    """Option to open dataset from existing CSV file or create from scratch.
    
    Rows are datapoints, columns are features. Last column of every row is y label (0 or 1).
    Simple features always come first, followed by complex features.
    """
    def __init__(self, gen=False, filename=None, **kwargs):
        if not gen:
            if filename != None:
                # Load datatype as float32 to avoid conflict with torch
                filepath = os.path.join('data', filename)
                self.dataset = np.genfromtxt(fname=filepath, delimiter=',', dtype="float32")
            else:
                raise FuncInputError
            self.num_points = self.dataset.shape[1]
            self.features = self.dataset.shape[0]
            self.simple = kwargs.pop('simple')
            self.complex = kwargs.pop('complex')

        else:
            for key in ['simple', 'complex', 'num_points']:
                if key not in kwargs:
                    raise KwargError

            self.num_points = kwargs.pop('num_points')
            self.simple = kwargs.pop('simple')
            # Array of n-point complexity for complex features
            self.complex_slabs = kwargs.pop('complex')
            # Num complex features
            self.complex = len(self.complex_slabs)
            self.features = self.simple + self.complex

            # Generate data
            self.generate()
            
    def split_randomise(self, x):
        """Randomise n-fractions of the dataset.
        
        Each subset is sensitive to one predictive feature.

        :param x: list of indices to be correlated with in each subset
        """
        n = len(x)  # Num of chunks
        chunk_size = self.__len__()//n  # Size of each chunk
        for i in range(n):
            idx = i*chunk_size
            np.random.shuffle(self.dataset[idx:idx+chunk_size, x[i]])
    
    def simple_randomise(self, frac):
        """Shuffle first fraction of dataset for whatever specified fraction."""
        k = int(frac*self.num_points)
        for i in range(self.simple):
            np.random.shuffle(self.dataset[:k, i])

    def generate(self):
        """"Generates dataset if it isn't read from file.
        
        :param complex: list of number of slabs to use for each feature
        """
        # Rows: datapoints
        # Cols: features
        self.dataset = np.empty((self.num_points, self.features+1), dtype="float32")

        for i in range(self.num_points):
            y = np.random.choice([0, 1], 1)
            # Last column is y label
            for j in range(self.simple):
                if y == 1:
                    self.dataset[i, j] = np.random.uniform(0, 1)
                elif y == 0:
                    self.dataset[i, j] = np.random.uniform(-1, 0)
            for j in range(self.complex):
                self.dataset[i, self.simple+j] = self.n_slabs(self.complex_slabs[j], y)
            self.dataset[i, -1] = y

    def n_slabs(self, n, y):
        """"Generate single x datapoint from single y label.
        
        Use k-point separation by dividing interval of length 2 into equally sized parts.

        :param n: number of slabs
        :param y: label
        """
        xs = np.linspace(-1, 1, n)
        if y == 1:
            x_poss = xs[::2]
            return np.random.choice(x_poss)
        elif y == 0:
            x_poss = xs[1::2]
            return np.random.choice(x_poss)
    
    def __len__(self):
        print(len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx):
        # Note that csv file is stored as cols = features, rows = datapoints
        input = self.dataset[idx, :-1]
        label = self.dataset[idx, -1]
        return input, label

# Train and test datasets formed slightly differently
def my_train_dataloader(gen=False, filename=None, simple=0, complex=[], num_points=0, mode=0, frac=0, x=[]):
    """Used to load train and/or test splits.
    
    :param gen: Binary variable stating whether data should be generated or loading from an existing file. If so, filename should be given by filename parameter.
    :param complex: List of complexity of k-slabs for every row that is a complex feature.
        E.g. [5,3] means we have a 5-point and 3-point complex feature.
    :param mode: Gives mode for data randomisation.
        0: Generates without any randomisation.
        1: Randomises feature indices/index listed in x using split randomise. Randomises simple feature entirely.
        Can use to randomise one specific complex feature at a time.
        2: Only randomises simple feature.
    :param x: List of indices (cols) for randomisation using split_randomise.
    :param frac: Fraction of simple coordinate to randomise (acts as noise on simple dataset)

    :returns: Dataset of 1D input features and one single label y which is the LAST COLUMN of the dataset.
    """
    # Base: to test that SB is observed as comparison
    dset = vecDataset(gen=gen, filename=filename, simple=simple, complex=complex, num_points=num_points)
    if gen:
        # Enforce variance towards all complex features only
        if mode == 1:
            if len(x) != 0:
                dset.split_randomise(x)
            dset.simple_randomise(frac)
        if mode == 2:
            dset.simple_randomise(frac)
        name = filename
        np.savetxt(output_dir+name, dset.dataset, delimiter=',')
    # Plot function for dataset visualisation
    #plot_data(dset.dataset, 1, 2)
    return (dset.dataset[:, :-1], dset.dataset[:, -1])

def my_test_dataloader(gen=False, filename=None, simple=0, complex=0, num_points=0,  sc=[]):
    """"Load test dataset.
    
    :param sc: subset of coordinates selected to randomise test distribution wrt to check if invariant
    """
    dset = vecDataset(gen=gen, filename=filename, simple=simple, complex=complex, num_points=num_points)
    if gen:
        for coord in sc:
            np.random.shuffle(dset.dataset[:, coord])
        name = filename
        np.savetxt(output_dir+name, dset.dataset, delimiter=',')
    #plot_data(dset.dataset, 1, 2)
    return (dset.dataset[:, :-1], dset.dataset[:, -1])

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        print('data shape: ', self.features.shape)
        print('labels shape: ', self.labels.shape)

    def __getitem__(self, index):
        # None adds extra dimension which hopefully forces dataloader to load correct size
        f = torch.tensor(self.features[index, :])
        l = torch.tensor(self.labels[index])
        return (f.to(device), l.to(device))

    def __len__(self):
        return len(self.labels)

# TESTING CODE
# GEN = True
# # Number of simple features
# NUM_SIMPLE = 1
# # Array defining number of slabs for each complex feature
# COMPLEX = [5, 8]
# # Total number of features
# NUM_FEATURES = NUM_SIMPLE + len(COMPLEX)
# # For train
# MODE = 1
# # Fraction of simple datapoints to randomise
# frac = 1
# X = [1]
# # For test - start with randomising simple feature (first row)
# SC = [0]
# FILE_TRAIN = 'train 1 1.csv'
# FILE_TEST = 'test 1 1.csv'
# NUM_POINTS = 3000
# X_train, y_train = my_train_dataloader(gen=GEN, filename=FILE_TRAIN, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, mode=MODE, frac=frac, x=X)
# X_test, y_test = my_test_dataloader(gen=GEN, filename=FILE_TEST, simple=NUM_SIMPLE, complex=COMPLEX, num_points=NUM_POINTS, sc=SC)