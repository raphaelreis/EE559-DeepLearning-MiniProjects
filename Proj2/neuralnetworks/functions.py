import math
import torch


Tensor = torch.Tensor

##################### Linear transformations #####################
def linear(input, weights, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    '''Apply a linear transformation to the input.'''
    
    output = input.matmul(weights.t())
    if bias is not None:
        output += bias
    return output



##################### Activation functions #####################
def tanh(input):
    # type: (Tensor) -> Tensor
    '''Hyperbolic tangeant'''

    return input.tanh()

def tanh_backward(dA, Z):
    # type: (Tensor, Tensor) -> Tensor
    '''Backward activation function for tanh'''

    tanh_squared = Z.tanh().pow(2)
    return (dA - dA.matmul(tanh_squared))

def relu(input):
    # type: (Tensor) -> Tensor
    '''Rectified linear unit'''

    return input.relu()

def relu_backward(dA, Z):
    # type: (Tensor, Tensor) -> Tensor
    '''Backward activation function for relu'''

    dZ = dA.clone()
    dZ[dZ <= 0] = 0
    return dZ


######################## Loss functions ########################
def mse(y_hat, y):
    '''Mean squared error'''

    return (y_hat - y).pow(2).sum()