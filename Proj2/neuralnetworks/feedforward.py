import logging
import math

import torch
from torch import empty

from .base import Module
from .functions import linear

torch.manual_seed(0)
log = logging.getLogger("TestMLP")


class Feedforward(Module):
    '''Fully connected neural network model'''
    def __init__(self, input_features, output_features, bias=True):
        super(Feedforward, self).__init__()
        self.init_parameters(input_features, output_features, bias)
        self.dl_dw = empty(output_features, input_features)
        self.delta = empty(output_features, 1)
        self.bias = bias

    def init_parameters(self, input_features, output_features, bias):
        self.W = kaimingHe_normal(output_features, input_features)
        if bias:
            self.b = empty(output_features, 1).zero_()

    def forward(self, x):
        self.input = x
        if self.bias:
            self.output = linear(x, self.W, self.b)
        else:
            self.output = linear(x, self.W)

    def backward(self, delta):
        self.delta = delta
        self.dl_dw = self.delta @ self.input.unsqueeze(-2)
        self.dl_db = self.delta
        return self.W.t() @ self.delta

    def update(self, lr):
        self.W = self.W - lr * self.dl_dw
        if self.bias:
            self.b = self.b - lr * self.delta

    def param(self):
        if self.bias:
            return [self.W, self.dl_dw, self.b, self.delta]
        else:
            return [self.W, self.dl_dw]

    def zero_grad(self):
        self.delta.zero_()
        self.dl_dw.zero_()


def kaimingHe_normal(output_size, input_size):
    std = math.sqrt(2. / (output_size))
    return empty(output_size, input_size).normal_(0., std)
