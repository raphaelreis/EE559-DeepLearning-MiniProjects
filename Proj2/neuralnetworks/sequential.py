
import logging

from .base import Module


###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################

log = logging.getLogger("TestMLP")


class Sequential(Module):
    def __init__(self):
        super(Module, self).__init__()
        self.mods = []

    def add(self, mod):
        self.mods.append(mod)

    def forward(self, input):
        for i, mod in enumerate(self.mods):
            if i > 0:
                mod.forward(self.mods[i-1].output)
            else:
                mod.forward(input)
        return self.mods[-1].output

    def backward(self, loss):
        delta = loss.derivate()
        for mod in reversed(self.mods):
            delta = mod.backward(delta)

    def param(self):
        params = []
        for mod in self.mods:
            if mod.param() != []:
                params.append(mod.param())
        return params

    def zero_grad(self):
        for mod in self.mods:
            if mod.param() != []:
                mod.zero_grad()
