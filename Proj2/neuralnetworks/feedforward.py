from torch import empty 

from .base import Module
from .functions import linear

class Feedforward(Module):
    '''Fully connected neural network model'''
    def __init__(self, input_features, output_features, bias=True):
        super(Feedforward, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = empty(input_features, output_features)
        if bias:
            self.bias = empty(output_features)
        self.init_parameters()  

    def init_parameters(self):
        self.weights.uniform_()
        if self.bias is not None:
            self.bias.uniform_()

    def forward(self, input):
        return linear(input, self.weights, self.bias)

    



