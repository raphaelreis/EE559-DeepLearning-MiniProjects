import math
import torch


Tensor = torch.Tensor

def linear(input, weights, bias=None):
    # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
    '''Apply a linear transformation to the input.'''
    
    output = input.matmul(weights.t())
    if bias is not None:
        output += bias
    return output

def tanh(input):
    # type: (Tensor) -> Tensor
    '''Hyperbolic tangeant'''

    return input.tanh()

def relu(input):
    # type: (Tensor) -> Tensor
    '''Rectified linear unit'''

    return input.relu()