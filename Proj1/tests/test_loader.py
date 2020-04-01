import sys
import unittest


import torch


from utils.loader import load




class TestLoadMethod(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def test_format(self):
        self.assertTrue(type(self.data) == tuple)
    
    def test_struct(self):
        for d in self.data:
            self.assertTrue(torch.is_tensor(d))
