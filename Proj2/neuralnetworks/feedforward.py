import logging

from torch import empty

from .base import Module
from .functions import linear, relu, relu_backward

log = logging.getLogger("TestMLP")


class Feedforward(Module):
    '''Fully connected neural network model'''
    def __init__(self, input_features, output_features,
                 activation='relu', bias=True):
        super(Feedforward, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.W = empty(input_features, output_features)
        if bias:
            self.b = empty(output_features)
        self.init_parameters()
        self.activation = self.parse_activation(activation)

    def init_parameters(self):
        self.W.uniform_()
        if self.b is not None:
            self.b.uniform_()

    def parse_activation(self, activation):
        if activation == 'relu':
            F = relu
            dF = relu_backward
        # elif activation == 'tanh':
        #     F = tanh
        #     dF = tanh_backward
        else:
            raise Exception("Not implemented activation function")

        d = dict(F=F, dF=dF)
        return d

    def forward(self, A):
        self.input = A
        self.Z = linear(A, self.W, self.b)

    def backward(self, dA_prev, dA_current):
        n = self.input.shape[0]
        dZ = self.activation['dF'](dA_current, self.Z)
        self.dW = (dA_prev.unsqueeze(1) @ dZ.unsqueeze(1).t()) / n
        self.db = (dZ / n).squeeze()
        self.dA_prev = (self.W @ dZ.unsqueeze(1)).squeeze()  

    def get_param(self):
        if not hasattr(self, 'dW') or not hasattr(self, 'db'):
            raise AssertionError("back propagation must be call first")
        
        return ((self.W, self.dW), (self.b, self.db))

    def set_param(self, W, b):
        self.W = W
        self.b = b
