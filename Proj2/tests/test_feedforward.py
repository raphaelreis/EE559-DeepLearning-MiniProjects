import sys
import logging
import unittest

from neuralnetworks.functions import MSE
from neuralnetworks.feedforward import Feedforward
from datagenerator.datagenerator import DataGenerator


logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger("TestFeedforward")


class TestPasses(unittest.TestCase):
    def testParameterInit(self):
        dg = DataGenerator(10)
        X_train, y_train, X_test, y_test = dg.get_data()
        loss = MSE()

        linear = Feedforward(2, 2)
        linear.forward(X_train[0])

        loss(linear.output, y_train[0])
        linear.backward(loss.derivate())
        param = linear.param()

        log.info("param init: {}".format(param))
