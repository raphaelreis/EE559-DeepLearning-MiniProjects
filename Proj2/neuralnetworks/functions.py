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

def relu(input):
    # type: (Tensor) -> Tensor
    '''Rectified linear unit'''

    return input.relu()


######################## Loss functions ########################
def mse(y_hat, y):
    '''Mean squared error'''
    return (y_hat - y).pow(2).sum()