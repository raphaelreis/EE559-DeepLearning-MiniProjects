import torch
import numpy as np

from .base import Module


class Dropout(Module):
    def __init__(self, p, input_size, seed=0):
        self.p = p
        self.generator = np.random.RandomState(seed)
        self.activation = self.generator.binomial(size=input_size, n=1, p=1-p)
        self.activation = torch.from_numpy(self.activation).float()
        self.train = True

    def set_training(self, b):
        self.train = b

    def forward(self, input):
        if self.train:
            self.output = input*self.activation
        else:
            self.output = input

    def backward(self, grad):
        return self.activation*grad
