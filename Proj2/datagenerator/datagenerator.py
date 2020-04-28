import math
from torch import empty

CERCLE_RADIUS = 1./math.sqrt(2.*math.pi)
DEFAULT_SAMPLESIZE = 1000


class DataGenerator:
    '''
        Train / test dataset generator for the inside of the 1/sqrt(pi) 
        radius cercle detector.
    '''
    def __init__(self, sample_size=DEFAULT_SAMPLESIZE):
        self.sample_size = sample_size

    def generate_data(self):
        # Tensor.uniform_() generates by default in the range [0, 1]
        # Train set
        self.X_train, self.y_train = self.gen_dat(self.sample_size)

        # Test set
        self.X_test, self.y_test = self.gen_dat(self.sample_size)

    def get_data(self):
        '''Get the generated data'''

        if not hasattr(self, 'X_train'):
            raise AssertionError("Need to run generate_data() function first")

        return self.X_train, self.y_train, self.X_test, self.y_test

    def gen_dat(self, sample_size):
        X = empty(self.sample_size, 2).uniform_()
        radii = X.pow(2).sum(axis=1).pow(1/2)
        y = (radii <= CERCLE_RADIUS).int()
        return X, y
