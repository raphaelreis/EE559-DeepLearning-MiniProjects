
from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, seq):
        for i, mod in enumerate(seq.mods):
            if (mod.param() != []):
                mod.update(self.lr)
    
    def zero_grad(self, seq):
        for mod in seq.mods:
            if mod.param() != []:
                mod.zero_grad()
