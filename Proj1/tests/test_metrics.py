import sys
import unittest
import logging

import torch

import utils.metrics as metrics

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class TestAccuracyFunction(unittest.TestCase):

    def testAssertion(self):
        a = torch.empty((5, 30))
        b = torch.empty((30, 1))
        
        with self.assertRaises(AssertionError):
            metrics.accuracy(a, b)

    def test1(self):
        a = torch.empty((20, 2)).normal_()
        b1 = a.argmax(1)
        b2 = a.argmin(1)
        log = logging.getLogger("TestAccuracyFunction")
        log.debug("Shape of a: {}".format(a.shape))
        
        self.assertEqual(1., metrics.accuracy(a, b1))
        self.assertEqual(0., metrics.accuracy(a, b2))