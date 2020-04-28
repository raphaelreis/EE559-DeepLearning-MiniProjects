import sys
import unittest
import logging

from torch import empty

from neuralnetworks.feedforward import Feedforward

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
log = logging.getLogger("TestFeedforward")


class TestPasses(unittest.TestCase):
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

