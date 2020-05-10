import sys
import unittest
import logging

from torch import empty

from neuralnetworks.mlp import MLP
from neuralnetworks.functions import one_hot
from datagenerator.datagenerator import DataGenerator

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger("TestMLP")


class TestMLP(unittest.TestCase):
    def testForward(self):
        X = empty(2)
        mlp = MLP(2, 2)
        mlp.forward(X)
        self.assertTrue(list(mlp.output.shape) == [2])

    def testBackward(self):
        X = empty(2)
        mlp = MLP(2, 2)
        mlp.forward(X)
        mlp.backward(mlp.output)
        log.debug("mlp.dInput.shape: {}".format(mlp.dInput.shape))
        self.assertTrue(list(mlp.dInput.shape) == [2])

    def testParameterSeter(self):
        X = empty(2)
        mlp = MLP(2, 2, 2)
        mlp.forward(X)
        mlp.backward(mlp.output)
        parameters = mlp.get_param()

        log.debug("mlp.get_param: \n{}".format(parameters[1][0][0]))
        new_weights = empty(2, 2).normal_()
        log.debug("new weights: {}".format(new_weights))
        # Create same set of parameters except second layer W
        new_parameters = []
        for layer in parameters:
            new_component = []
            for comp in layer:
                new_type = []
                for ty in comp:
                    if ty.eq(parameters[1][0][0]).all().item():
                        new_type.append(new_weights)
                    else:
                        new_type.append(ty)
                new_component.append(tuple(new_type))
            new_parameters.append(tuple(new_component))

        log.debug("new_paramters: {}".format(new_parameters[1][0][0]))
        mlp.set_param_to_layer(0, new_parameters[0][0][0],
                               new_parameters[0][1][0])
        mlp.set_param_to_layer(1, new_parameters[1][0][0],
                               new_parameters[1][1][0])

        set_paramters = mlp.get_param()
        log.debug("new_paramters: {}".format(set_paramters[1][0][0]))
        self.assertTrue(set_paramters[1][0][0].eq(new_weights).all().item())

    def testTrainSimple(self):
        X = empty(100, 2).normal_()
        y = empty(100).normal_()
        mlp = MLP(2, 1)
        mlp.forward(X[0])
        log.debug("mlp.forward(X[0]) -> output: {}".format(mlp.output))
        log.debug("groud truth: {}".format(y[0]))
        mlp.train(X, y, 10)
        mlp.forward(X[0])
        log.debug("mlp.forward(X[0]) -> output: {}".format(mlp.output))
        log.debug("ground truth: {}".format(y[0]))

    def testTrainOneHot(self):
        dg = DataGenerator(100)
        dg.generate_data()
        X_train, y_train, _, _ = dg.get_data()
        y_train_hot = one_hot(y_train)

        mlp = MLP(2, 2)

        mlp.forward(X_train[1])
        log.debug("mlp.forward(X_train[0]) -> output: {}".format(mlp.output))
        log.debug("ground truth: {}".format(y_train[0]))
        mlp.train(X_train, y_train_hot, 10)

        mlp.forward(X_train[1])
        log.debug("mlp.forward(X[0]) -> output: {}".format(mlp.output))
        log.debug("ground truth: {}".format(y_train[0]))

    def testWithDataGenerator(self):
        dg = DataGenerator(10)
        dg.generate_data()
        X_train, y_train, X_test, y_test = dg.get_data()

        mlp = MLP(2, 1)
        mlp.train(X_train, y_train.float(), 1)

    def testTestFunction(self):
        dg = DataGenerator(1000)
        dg.generate_data()
        X_train, y_train, X_test, y_test = dg.get_data()

        mlp = MLP(2, 2)
        mlp.train(X_train, y_train, 1)
        accuracy = mlp.test(X_test, y_test)
        log.debug("final Accuracy: {}".format(accuracy))
