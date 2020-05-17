import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

###################################################
#           Le_Net inspired network               #
#     Use weight sharing over the two channels    #
#Use auxiliary loss that classify the digit number#
###################################################


class LeNet_sharing_aux(nn.Module):
    """
    Weight sharing + Auxiliary loss
    
    """
    def __init__(self,nbhidden_aux = 200,nbhidden_comp=60,drop_prob_aux = 0.2,drop_prob_comp = 0):
        super(LeNet_sharing_aux, self).__init__()
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, nbhidden_aux)
        self.fc2 = nn.Linear(nbhidden_aux, 10)
        # Dropout 
        self.dropout_aux = nn.Dropout(drop_prob_aux)
        self.dropout_comp = nn.Dropout(drop_prob_comp)
        
        # weights for binary classification 
        self.fc3 = nn.Linear(20, nbhidden_comp)
        self.fc4 = nn.Linear(nbhidden_comp, 100)
        self.fc5 = nn.Linear(100, 2)
        
    def forward(self, input_):    
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = self.dropout_aux(x)
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout_aux(x)
        x = self.fc2(x)
        
        # forward pass for the second image 
        y = F.relu(F.max_pool2d(self.bn1(self.conv1(y)), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2))
        y = self.dropout_aux(y)
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.dropout_aux(y)
        y = self.fc2(y)
        
        # concatenate layers  
        z = torch.cat([x, y], 1)
        
        # Binary classification
        z = F.relu(self.fc3(z))
        z = F.relu(self.dropout_comp(self.fc4(z)))
        z = self.dropout_comp(self.fc5(z))
        
        return x, y, z
    
##################################################
#           Le_Net inspired network              #
#     Use weight sharing over the two channels   #
##################################################

class LeNet_sharing(nn.Module):
    """
    Weight sharing 
    
    """
    def __init__(self, nb_hidden = 100, dropout_ws = 0,dropout_comp = 0):
        super(LeNet_sharing, self).__init__()
        
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        # fully connected layers 
        self.fc3 = nn.Linear(20, 60)
        self.fc4 = nn.Linear(60, 90)
        self.fc5 = nn.Linear(90, 2)
        # dropout proba
        self.dropout_ws = nn.Dropout(dropout_ws)
        self.dropout_comp = nn.Dropout(dropout_comp)
        
    def forward(self, input_):        
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.dropout_ws(x)
        x = self.fc2(x)
        # forward pass for the second image
        y = F.relu(F.max_pool2d(self.conv1(y), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.conv2(y), kernel_size=2, stride=2))
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.dropout_ws(y)
        y = self.fc2(y)
        
        # concatenate layers 
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.dropout_comp(z)
        z = self.fc5(z)
        
        return  z