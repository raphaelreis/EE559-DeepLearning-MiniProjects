import logging

from .base import Module
from .feedforward import Feedforward
from .functions import tanh, MSE, MSE_prime
# from ..optimizer.sgd import SGD


###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################

log = logging.getLogger("TestMLP")


class MLP(Module):
    def __init__(self, feature_size, output_size, hidden_size=25, optim='SGD',
                 lr=0.01, activation='relu', loss='MSE'):
        super(MLP, self).__init__()
        # if optim is None:
        #     raise TypeError("No optimizer given")
        
        # def parse_optim(self, optim):
        #     if optim == 'SGD':
        #         return SGD
        #     else:
        #         raise Exception("Not implemented optimizer function")

        def parse_loss(loss):
            if loss == 'MSE':
                return MSE, MSE_prime
            else:
                raise Exception("Not implemented loss function")

        # self.optimizer = parse_optim(optim)
        self.lr = lr
        self.optimizer = optim
        self.loss, self.dLoss = parse_loss(loss)
        self.linear1 = Feedforward(feature_size,
                                   hidden_size, activation=activation)
        self.linear2 = Feedforward(hidden_size,
                                   output_size, activation=activation)
        self.layers = [self.linear1, self.linear2]

    def forward(self, X: Tensor):
        self.linear1.forward(X)
        x = self.linear1.Z
        self.linear2.forward(x)
        self.output = tanh(self.linear2.Z)

    def backward(self, dA: Tensor):
        self.linear2.backward(self.linear2.input, dA)
        dx = self.linear2.dA_prev
        self.linear1.backward(self.linear1.input, dx)
        self.dInput = self.linear1.dA_prev

    def get_param(self):
        return [self.linear1.get_param(), self.linear2.get_param()]

    def set_param_to_layer(self, idx, W, b):
        self.layers[idx].set_param(W, b)

    def train(self, X, y, epochs=10):
        # self.optimizer()
        if self.optimizer == 'SGD':
            for e in range(epochs):
                for (dat, tar) in zip(X, y):
                    self.forward(dat)
                    dL = self.dLoss(self.output, y)
                    self.backward(dL)

                    # Update
                    for idx, ((W, dW), (b, db)) in enumerate(self.get_param()):
                        # log.debug("W: {}".format(W))
                        # log.debug("dW: {}".format(dW))
                        W_t_1 = W - self.lr * dW
                        b_t_1 = b - self.lr * db
                        self.set_param_to_layer(idx, W_t_1, b_t_1)




