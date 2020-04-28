import torch 
import sys
sys.path.append('..')
from torch import nn 
from torch.nn import functional as F
from torch import optim

#######################
### BASELINE MODELS ###
#######################

class Net2C(nn.Module):
    
    """
    Network which takes input as a two channel 14*14 image
    
    """

    def __init__(self, nb_hidden):
        
        super(Net2C, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 5) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 2) 
        
    def forward(self, x):
        
        # forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 3, stride = 1))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 3, stride = 3))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
    
        return x 
    
    
class Netcat(nn.Module):
    
    """
    Network which processes the input to get a  : 1000 * 1 * 14 * 28 set 
    
    """
    
    def __init__(self, dim):
        
        super(Netcat, self).__init__()
        self.d = dim
        self.conv1 = nn.Conv2d(1, 6, kernel_size = 3, stride = 1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size = 3, stride = 1)
        self.fc1 = nn.Linear(480, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)   
  
        
    def forward(self, x):
   
        x = torch.cat((x[:,0], x[:,1]), dim = 2).unsqueeze(dim = self.d)   # concatenate channels into 1 channel (input : 1000 * 1 * 14 * 28)
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size = 2, stride = 2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size = 2, stride = 1))
        x = F.relu(self.fc1(x.view(-1, 480)))
        x = F.relu(self.fc2(x.view(-1, 120)))
        x = self.fc3(x)

        return x
    
    
#############################
###### WEIGHT SHARING #######
############################# 

class LeNet_WS_sequential(nn.Module):
    """
    Weight sharing 
    
    """
    def __init__(self, nb_hidden):
        super(LeNet_WS_sequential, self).__init__()
        
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        # fully connected layers 
        self.fc3 = nn.Linear(20, 60)
        self.fc4 = nn.Linear(60, 90)
        self.fc5 = nn.Linear(90, 2)
        
    def forward(self, input_):        
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        # forward pass for the second image
        y = F.relu(F.max_pool2d(self.conv1(y), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.fc2(y)
        
        # concatenate layers 
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        
        return  z
    
    
###############################
###### AUXILIARY LOSS #########
###############################

class LeNet_aux_sequential(nn.Module):
    """
    Weight sharing + Auxiliary loss
    
    """
    def __init__(self):
        super(LeNet_aux_sequential, self).__init__()
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, 10)
        
        # weights for binary classification 
        self.fc3 = nn.Linear(20, 60)
        self.fc4 = nn.Linear(60, 90)
        self.fc5 = nn.Linear(90, 2)
        
    def forward(self, input_):    
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        
        # forward pass for the second image 
        y = F.relu(F.max_pool2d(self.conv1(y), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.fc2(y)
        
        # concatenate layers  
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        
        return x, y, z