
import logging

from .base import Module
from .feedforward import Feedforward
from .functions import tanh, MSE, MSE_prime, sigmoid
# from ..optimizer.sgd import SGD


###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################

log = logging.getLogger("TestMLP")


class MLP(Module):
    def __init__(self, feature_size, output_size, hidden_size=25, optim='SGD',
                 lr=0.01, activation='relu', last_layer_activation='tanh',
                 loss='MSE'):
        super(MLP, self).__init__()

        def parse_loss(loss):
            if loss == 'MSE':
                return MSE, MSE_prime
            else:
                raise Exception("Not implemented loss function")

        def parse_last_layer_activation(last_layer_activation):
            if last_layer_activation == 'sigmoid':
                return sigmoid
            elif last_layer_activation == 'tanh':
                return tanh
            else:
                raise Exception("Not implemented activation function")

        self.lr = lr
        self.optimizer = optim
        self.loss, self.dLoss = parse_loss(loss)
        self.linear1 = Feedforward(feature_size,
                                   hidden_size, activation=activation)
        self.linear2 = Feedforward(hidden_size,
                                   output_size, activation=activation)
        self.last_layer_activation = parse_last_layer_activation(
                                        last_layer_activation)
    
    def forward(self, X: Tensor):
        self.linear1.forward(X)
        x = self.linear1.Z
        self.linear2.forward(x)
        # log.debug("linear2.Z: {}".format(self.linear2.Z))
        self.output = self.last_layer_activation(self.linear2.Z)

    def backward(self, dA: Tensor):
        self.linear2.backward(self.linear2.input, dA)
        dx = self.linear2.dA_prev.squeeze()
        self.linear1.backward(self.linear1.input, dx)
        self.dInput = self.linear1.dA_prev

    def get_param(self):
        return [self.linear1.get_param(), self.linear2.get_param()]

    def set_param_to_layer(self, idx, W, b):
        if idx == 0:
            self.linear1.set_param(W, b)
        else:
            self.linear2.set_param(W, b)

    def train(self, X, y, epochs=10):
        history_loss = []
        if self.optimizer == 'SGD':
            for e in range(epochs):
                correct = 0
                for (dat, tar) in zip(X, y):
                    self.forward(dat)
                    loss = self.loss(self.output, tar)
                    history_loss.append(loss.item())

                    # log.debug("self.output: {}".format(self.output))
                    # log.debug("tar: {}".format(tar))
                    # log.debug("loss: {}".format(loss))

                    dL = self.dLoss(self.output, tar)
                    self.backward(dL)

                    for idx, ((W, dW), (b, db)) in enumerate(self.get_param()):
                        W_t_1 = (W - self.lr * dW).clone()
                        b_t_1 = (b - self.lr * db).clone()
                        self.set_param_to_layer(idx, W_t_1, b_t_1)
                    
                    self
                    
        else:
            raise NotImplementedError("There is currently no other optimizer")

        self.train_history_loss = history_loss
