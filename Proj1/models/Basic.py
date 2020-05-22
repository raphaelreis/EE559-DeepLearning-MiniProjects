import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

########################################################################
#                Simple Convolutional Neural Network (CNN)             #
#                Inspired from the NN in Practical 4 (EE-559)          #           
#                No weights sharing and  auxiliary losses              #
########################################################################

class Net2C(nn.Module):
    
    """
    Input : Nx2x14x14 -> N pairs of digits (14x14) in a two-channels image
    Output : Nx2 -> N tensor of two features
    
    Parameters :
    
    1) nb_hidden : number of nodes of the hidden layer
    2) dropout_prob : dropout rate of the dropout module 
    
    """

    def __init__(self, nb_hidden,dropout_prob):
        
        super(Net2C, self).__init__()
        
        # Convolutional layers  
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        
        # Linear layers
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2) 
        
        # Dropout 
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        
        # forward pass -> convolutinal layers are followed by max pooling and Relu activation function
        #              -> Dropout used on the the nodes of the hidden layer 
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1)) # Nx32x8x8
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3)) # Nx64x2x2
        x = F.relu(self.fc1(x.view(-1, 256))) # Nxnb_hidden
        x = self.dropout(x) # Nxnb_hidden -> some element are put randomly to zero by dropout
        x = self.fc2(x) # Nx2
    
        return x

###########################################################################################################