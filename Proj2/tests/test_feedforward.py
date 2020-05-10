import sys
import unittest
import logging

from torch import empty

from neuralnetworks.feedforward import Feedforward, kaiming
from neuralnetworks.functions import relu

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger("TestFeedforward")


class TestFunctions(unittest.TestCase):
    def testKaiming(self):
        input = 25
        output = 25
        x = empty(input).normal_()
        for i in range(100):
            a = kaiming(input, output)
            x = relu(a @ x)


class TestPasses(unittest.TestCase):
    def testSetParameters(self):
        linear = Feedforward(2, 2)
        new_weights = empty(2, 2).normal_()
        new_bias = empty(2).normal_()
        linear.set_param(new_weights, new_bias)
        
        self.assertTrue(linear.W.eq(new_weights).all().item())
        self.assertTrue(linear.b.eq(new_bias).all().item())

    def testSimpleForward(self):
        X = empty(2).normal_()
        linear = Feedforward(2, 2)
        linear.forward(X)

        self.assertTrue(list(linear.Z.shape) == [2],
                        "Simple forward path does not work")

    def testForwardInduction(self):
        X = empty(2).normal_()
        linear1 = Feedforward(2, 25)
        linear1.forward(X)
        linear2 = Feedforward(25, 2)
        linear2.forward(linear1.Z)

        self.assertTrue(list(linear2.Z.shape) == [2],
                        "Forward induction does not work")

    def testSimpleBackward(self):
        X = empty(2).normal_()

        linear = Feedforward(2, 2)
        linear.forward(X)
        linear.backward(linear.Z)

        log.info("linear.dA_prev: {}".format(linear.dA_prev))
        self.assertTrue(list(linear.dA_prev.shape), [2])

    def testBackwardInduction(self):
        X = empty(2).normal_()
        # Forward
        linear1 = Feedforward(2, 25)
        linear1.forward(X)
        linear2 = Feedforward(25, 2)
        linear2.forward(linear1.Z)
        # Backward
        linear2.backward(linear2.Z)
        linear1.backward(linear2.dA_prev)

        log.info("linear1.dA_prev: {}".format(linear1.dA_prev))
        self.assertTrue(list(linear1.dA_prev.shape), [2])

