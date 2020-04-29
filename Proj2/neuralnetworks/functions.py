###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################

##################### Linear transformations ##################### noqa: E266


def linear(input: Tensor, weights: Tensor, bias: Tensor = None) -> Tensor:

    '''Apply a linear transformation to the input.'''

    output = input.t() @ weights
    if bias is not None:
        output += bias
    return output


##################### Activation functions ##################### noqa: E266


def tanh(input: Tensor) -> Tensor:
    '''Hyperbolic tangent'''

    return input.tanh()


def sigmoid(input: Tensor) -> Tensor:
    '''Sigmoid function'''

    return input.sigmoid()

# def tanh_backward(dA: Tensor, Z: Tensor) -> Tensor:
#     '''Backward activation function for tanh'''

#     tanh_squared = Z.tanh().pow(2)
#     return (dA - dA.matmul(tanh_squared))


def relu(input: Tensor) -> Tensor:
    '''Rectified linear unit'''

    return input.relu()


def relu_backward(dA: Tensor, Z: Tensor) -> Tensor:
    '''Backward activation function for relu'''

    dZ = dA.clone()
    dZ[dZ <= 0] = 0
    return dZ


######################## Loss functions ######################## noqa: E266

def MSE(y_hat: Tensor, y: Tensor) -> Tensor:
    '''Mean squared error'''

    return (y_hat - y).pow(2).sum()


def MSE_prime(y_hat: Tensor, y: Tensor) -> Tensor:
    '''Derivative of the mean squared error'''

    return -2 * (y_hat - y)

############################ Utils ############################# noqa: E266


def one_hot(y, dims=2):
    y_hot = y.long().view(-1, 1).clone()
    y_one_hot = torch.empty(y_hot.size()[0], dims).\
        fill_(0.).scatter_(1, y_hot, 1)
    return y_one_hot
