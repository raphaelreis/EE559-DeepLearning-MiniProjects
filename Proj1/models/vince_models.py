import torch 
import sys

sys.path.append('..')

from torch import nn 
from torch.nn import functional as F
from torch import optim


# convolutional network for binary classification (1 output unit)


class Net2(nn.Module):

    def __init__(self, nb_hidden):
        super(Net2, self).__init__()
        
        # number of input channels is 2.
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 1) # single output
        self.sigmoid = nn.Sigmoid()  
        
    def forward(self,x):
    
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        x = self.sigmoid(x) 
        return x # returns a probability 






