import sys
import random
import unittest
import logging

from neuralnetworks.sequential import Sequential
from neuralnetworks.feedforward import Feedforward
from neuralnetworks.functions import ReLU, Tanh, MSE
from datagenerator.datagenerator import DataGenerator
from optimizer.sgd import SGD

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger("TestMLP")


class TestMLP(unittest.TestCase):

    def testWeightsUpdate(self):
        dg = DataGenerator(1000)
        X_train, y_train, X_test, y_test = dg.get_data()

        mlp = Sequential()
        mlp.add(Feedforward(2, 2))
        mlp.add(ReLU())
        mlp.add(Feedforward(2, 2))
        mlp.add(Tanh())
        loss = MSE()
        optimizer = SGD(0.01)

        output = mlp.forward(X_train[0])
        loss(output, y_train[0])
        mlp.backward(loss)
        optimizer.step(mlp)

        before_train_param = mlp.param()

        output = mlp.forward(X_train[0])
        loss(output, y_train[0])
        mlp.backward(loss)
        optimizer.step(mlp)

        after_train_param = mlp.param()
        self.assertFalse(before_train_param[0][0]
                         .eq(after_train_param[0][0]).all().item())
        self.assertFalse(before_train_param[0][2]
                         .eq(after_train_param[0][2]).all().item())
        self.assertFalse(before_train_param[1][0]
                         .eq(after_train_param[1][0]).all().item())
        self.assertFalse(before_train_param[1][2]
                         .eq(after_train_param[1][2]).all().item())

    def testZeroGrad(self):
        '''Test that the zero_grad method anihilate the gradient'''

        dg = DataGenerator(1000)
        X_train, y_train, X_test, y_test = dg.get_data()

        mlp = Sequential()
        mlp.add(Feedforward(2, 2))
        mlp.add(Tanh())
        mlp.add(Feedforward(2, 2))
        mlp.add(Tanh())
        lr = 0.01
        optimizer = SGD(lr)
        loss = MSE()

        idx = random.randint(0, X_train.shape[0]-1)
        val, tar = X_train[idx], y_train[idx]
        output = mlp.forward(val)
        loss(output, tar)
        mlp.backward(loss)

        optimizer.step(mlp)
        log.info("mlp.param()[0][1]: {}".format(mlp.param()[0][1]))
        log.info("mlp.param()[0][3]: {}".format(mlp.param()[0][3]))
        optimizer.zero_grad(mlp)
        log.info("mlp.param()[0][1]: {}".format(mlp.param()[0][1]))
        log.info("mlp.param()[0][3]: {}".format(mlp.param()[0][3]))
        self.assertEqual((mlp.param()[0][1] == 0.).int().sum().item(), 4)
        self.assertEqual((mlp.param()[0][3] == 0.).int().sum().item(), 2)
