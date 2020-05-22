
from .base import Module

###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################

##################### Linear transformations ##################### noqa: E266


def linear(input: Tensor, weights: Tensor, bias: Tensor = None) -> Tensor:

    '''Apply a linear transformation to the input.'''

    output = weights @ input
    if bias is not None:
        output.add(bias)
    return output


##################### Activation functions ##################### noqa: E266


def tanh(input: Tensor) -> Tensor:
    '''Hyperbolic tangent'''

    return input.tanh()


def d_tanh(input: Tensor) -> Tensor:
    '''Hyperbolic angent derivative'''

    return 1 - input.tanh().pow(2)


def sigmoid(input: Tensor) -> Tensor:
    '''Sigmoid function'''

    return input.sigmoid()


def d_sigmoid(x: Tensor) -> Tensor:
    return (-x).exp() / (1 + (-x).exp()).pow(2)


def relu(x):
    return x * (x > 0).float()


def d_relu(x):
    return 1. * (x > 0).float()

######################## Activation modules ######################## noqa: E266


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.act = relu
        self.d_act = d_relu

    def forward(self, x):
        self.input = x
        self.output = self.act(x)

    def backward(self, delta_network):
        delta_activation = delta_network * self.d_act(self.input).view(-1, 1)
        return delta_activation


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.act = sigmoid
        self.d_act = d_sigmoid

    def forward(self, x):
        self.input = x
        self.output = self.act(x)

    def backward(self, delta_network):
        delta_activation = delta_network * self.d_act(self.input).view(-1, 1)
        return delta_activation


class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.act = tanh
        self.d_act = d_tanh

    def forward(self, x):
        self.input = x
        self.output = self.act(x)

    def backward(self, delta_network):
        delta_activation = delta_network * self.d_act(self.input).view(-1, 1)
        return delta_activation

######################## Loss function ######################## noqa: E266


def mse(y_hat: Tensor, y: Tensor) -> Tensor:
    '''Mean squared error'''

    return (y_hat - y).pow(2).sum()


def mse_prime(y_hat: Tensor, y: Tensor) -> Tensor:
    '''Derivative of the mean squared error'''

    return 2 / y.shape[0] * (y_hat - y)


class MSE:
    def __init__(self):
        self.loss = mse
        self.derivative = mse_prime

    def __call__(self, output, target):
        self.output = output
        self.target = target
        self.value = self.loss(output, target)

    def derivate(self):
        return self.derivative(self.output, self.target).view(-1, 1)
