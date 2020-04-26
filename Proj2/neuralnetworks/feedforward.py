from torch import empty 

from .base import Module
from .functions import linear, relu, tanh, mse

class Feedforward(Module):
    '''Fully connected neural network model'''
    def __init__(self, input_features, output_features, 
                    activation='relu', loss='mse', bias=True):
        super(Feedforward, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weights = empty(input_features, output_features)
        if bias:
            self.bias = empty(output_features)
        self.init_parameters()
        self.activation = parse_activation(activation)
        self.loss = parse_loss(loss)
    def init_parameters(self):
        self.weights.uniform_()
        if self.bias is not None:
            self.bias.uniform_()

    def parse_activation(activation):
        if activation == 'relu':
            F = relu
            dF = relu_backward
        elif activation == 'sigmoid':
            F = tanh
            dF = tanh_backward
        else:
            raise Exception("Not implemented activation function")
        
        d = dict(F=F, dF=dF)
        return d

    def parse_loss(loss):
        if loss == 'mse':
            return mse
        else:
            raise Exception("Not implemented loss function")

    def forward(self, input):
        return linear(input, self.weights, self.bias)

    def backward(self, )
            
        


    



