

###### Only for intellisense ###### noqa: E266
import torch
Tensor = torch.Tensor
##################################


class Optimizer:
    def step(self, seq):
        raise NotImplementedError
