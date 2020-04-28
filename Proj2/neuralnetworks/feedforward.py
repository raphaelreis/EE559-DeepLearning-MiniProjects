import logging

from torch import empty

from .base import Module
from .functions import linear, relu, relu_backward, tanh, tanh_backward

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
        log.debug("\n*****BACKWARD*****")
        log.debug("dA_prev: {}".format(dA_prev))
        log.debug("dA_current: {}".format(dA_current))
        n = self.input.shape[0]

        dZ = self.activation['dF'](dA_current, self.Z)
        log.debug("dZ shape: {}".format(dZ.shape))
        log.debug("dA_prev.shape: {}".format(dA_prev.shape))
        self.dW = (dA_prev.unsqueeze(1) @ dZ.unsqueeze(1).t()) / n
        log.debug("dW: {}".format(self.dW.shape))
        self.db = (dZ / n).squeeze()
        # log.debug("db: {}".format(self.db))
        log.debug("W shape: {}".format(self.W.shape))
        # log.debug("dZ shape: {}".format(dZ.shape))
        self.dA_prev = (self.W @ dZ.unsqueeze(1)).squeeze()
        # log.debug("dA_prev: {}".format(self.dA_prev))

    def get_param(self):
        return ((self.W, self.dW), (self.b, self.db))

    def set_param(self, W, b):
        self.W = W
        self.b = b
