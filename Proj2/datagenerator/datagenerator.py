import math

import torch
from torch import empty

from .utils import one_hot

CERCLE_RADIUS = 1./math.sqrt(2.*math.pi)
DEFAULT_SAMPLESIZE = 1000


class DataGenerator:
    '''
        Train / test dataset generator for the inside of the 1/sqrt(pi) 
        radius cercle detector.
    '''
    def __init__(self, sample_size=DEFAULT_SAMPLESIZE):
        self.sample_size = sample_size
        self.X_train, self.y_train = self.gen(self.sample_size)
        self.X_test, self.y_test = self.gen(self.sample_size)

    def gen(self, sample_size):
        '''Generate the data with specified constrains'''
        X = empty(self.sample_size, 2).uniform_()
        radii = (X-torch.ones(self.sample_size, 2) * 0.5).abs()\
            .pow(2).sum(axis=1).pow(1/2)
        y = (radii <= CERCLE_RADIUS).int()
        return X, y

    def get_data(self, oh=True):
        '''Get the generated data'''

        if oh:
            return self.X_train, one_hot(self.y_train),\
                self.X_test, one_hot(self.y_test)
        else:
            return self.X_train, self.y_train, self.X_test, self.y_test
