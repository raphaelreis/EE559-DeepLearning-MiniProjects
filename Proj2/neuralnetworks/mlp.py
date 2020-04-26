from .base import Module
from .feedforward import Feedforward
from .functions import tanh


###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################


class MLP(Module):
    def __init__(self, feature_size, output_size, activation='relu'):
        super(MLP, self).__init__()
        self.linear1 = Feedforward(feature_size, 25, activation=activation)
        self.linear2 = Feedforward(25, output_size, activation=activation)

    def forward(self, X: Tensor):
        self.linear1.forward(X)
        x = self.linear1.Z
        self.linear2.forward(x)
        self.output = tanh(self.linear2.Z)

    def backward(self, dA: Tensor):
        self.linear1.backward(dA)
        dx = self.linear1.dA_prev
        self.linear2.backward(dx)
        self.dInput = self.linear2.dA_prev

    def param(self):
        return []
