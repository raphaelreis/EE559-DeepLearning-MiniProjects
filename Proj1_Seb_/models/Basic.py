import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

####################################################
#           Simple binary classifiers              #
#                 Baseline Models                  #
#              No WS / No Aux Loss                 #
####################################################

class Net2C(nn.Module):
    
    """
    Simple Network which takes input as a two channel 14*14 image 
    
    """

    def __init__(self, nb_hidden,dropout_prob):
        
        super(Net2C, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2) 
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        
        # forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout(x)
        x = self.fc2(x)
    
        return x 

    
class Netcat(nn.Module):
    
    """
    Network which processes the input to get a  : 1000 * 1 * 14 * 28 set by concantenating the twi images in each pair
    
    """
    
    def __init__(self):
        
        super(Netcat, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)   
  
        
    def forward(self, x):
   
        x = torch.cat((x[:,0], x[:,1]), dim = 2).unsqueeze(dim = 1)   # concatenate channels into 1 channel (input : 1000 * 1 * 14 * 28)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        x = F.relu(self.fc1(x.view(-1, 480)))
        x = F.relu(self.fc2(x.view(-1, 120)))
        x = self.fc3(x)

        return x