import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader

###################################################################
#                    Le_Net inspired network                      #
#        Same CNN applied on each channel -> weight sharing       # 
#        Auxiliary loss on the output of the shared CNN           #
#    Resulting features over the two channels are concatenated    #
#          Linear layers over the concatenated result             #
###################################################################


class LeNet_sharing_aux(nn.Module):
    """
    General :
        Input : Nx2x14x14 -> N pairs of digits (14x14) in a two-channels image
        Output : Nx10, Nx10 -> tensor of two features for digit classification, Nx2 -> N tensor of two features for binary classification

    CNN over each channel :
        Input : Nx1x14x14
        Output :Nx10
    Concatenated output CNN :
        [Nx10,Nx10] -> Nx20
    FC layers over the Concatenated output of the CNN  : 
        Input : Nx20
        output : Nx2
    
    Parameters : 
        1) nb_hidden_aux : number of nodes of the last layer of the CNN's FC layers
        2) nb_hidden_comp : number of nodes of the second layer of the FC layers
        3) drop_prob_aux : dropout rate of the dropout module used in the CNN
        4) drop_prob_comp : dropout rate on the  elements of the last layer of the FC layers 
    
    """
    def __init__(self,nbhidden_aux = 200,nbhidden_comp=60,drop_prob_aux = 0.2,drop_prob_comp = 0):
        super(LeNet_sharing_aux, self).__init__()
        
        # convolutional layers of the shared CNN with batch norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        
        # FC layers of the shared CNN
        self.fc1 = nn.Linear(256, nbhidden_aux)
        self.fc2 = nn.Linear(nbhidden_aux, 10)
        
        # Dropout 
        self.dropout_aux = nn.Dropout(drop_prob_aux)
        self.dropout_comp = nn.Dropout(drop_prob_comp)
        
        # fully connected layers 
        self.fc3 = nn.Linear(20, nbhidden_comp)
        self.fc4 = nn.Linear(nbhidden_comp, 100)
        self.fc5 = nn.Linear(100, 2)
        
    def forward(self, input_):    
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2)) # Nx32x6x6
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2)) # Nx64x2x2
        x = self.dropout_aux(x) # Nx64x2x2 -> some element are put randomly to zero by dropout
        x = F.relu(self.fc1(x.view(-1, 256))) # Nxnb_hidden_aux
        x = self.dropout_aux(x) # Nxnb_hidden_aux -> some element are put randomly to zero by dropout
        x = self.fc2(x) # Nx10
        
        # forward pass for the second image 
        y = F.relu(F.max_pool2d(self.bn1(self.conv1(y)), kernel_size=2, stride=2)) # Nx32x6x6
        y = F.relu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2)) # Nx64x2x2
        y = self.dropout_aux(y) # Nx64x2x2 -> some element are put randomly to zero by dropout
        y = F.relu(self.fc1(y.view(-1, 256))) # Nxnb_hidden_aux
        y = self.dropout_aux(y) # Nxnb_hidden_aux -> some element are put randomly to zero by dropout
        y = self.fc2(y) # Nx10
        
        # concatenate layers  
        z = torch.cat([x, y], 1) #Nx20
        
        # FC layer for binary classification 
        z = F.relu(self.fc3(z)) # Nx20
        z = self.dropout_comp(z)  # Nx20 -> some element are put randomly to zero by dropout
        z = F.relu(self.fc4(z))  # Nxnbhidden_comp
        z = self.dropout_comp(z)  # Nxnbhidden_comp -> some element are put randomly to zero by dropout
        z = self.fc5(z) #Nx2
        
        return x, y, z
    
###################################################################
#                    Le_Net inspired network                      #
#        Same CNN applied on each channel -> weight sharing       # 
#    Resulting features over the two channels are concatenated    #
#          Linear layers over  the concatenated result            #
###################################################################

class LeNet_sharing(nn.Module):
    """
    General :
        Input : Nx2x14x14 -> N pairs of digits (14x14) in a two-channels image
        Output : Nx2 -> N tensor of two features

    CNN over each channel :
        Input : Nx1x14x14
        Output :Nx10
    Concatenated output CNN :
        [Nx10,Nx10] -> Nx20
    FC layers over the Concatenated output of the CNN  : 
        Input : Nx20
        output : Nx2
    
    Parameters : 
        1) nb_hidden : number of nodes of the last layer of the CNN's FC layers
        2) dropout_ws : dropout rate of the dropout module used in the CNN
        3) dropout_comp : dropout rate on the  elements of the last layer of the FC layers
    
    """
    def __init__(self, nb_hidden = 100, dropout_ws = 0,dropout_comp = 0):
        super(LeNet_sharing, self).__init__()
        
        # convolutional layers of the shared CNN with batch norm
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        # FC layers of the shared CNN
        self.fc1 = nn.Linear(256, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)
        
        # fully connected layers 
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 2)
        
        # dropout
        self.dropout_ws = nn.Dropout(dropout_ws)
        self.dropout_comp = nn.Dropout(dropout_comp)
        
    def forward(self, input_):        
        
        # split the 2-channel input into two 14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # forward pass for the first image 
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2)) # Nx32x6x6
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2)) # Nx64x2x2
        x = self.dropout_ws(x) # Nx64x2x2 -> some element are put randomly to zero by dropout
        x = F.relu(self.fc1(x.view(-1, 256))) # Nxnb_hidden
        x = self.dropout_ws(x) # Nxnb_hidden -> some element are put randomly to zero by dropout
        x = self.fc2(x) # Nx10
        
        # forward pass for the second image
        y = F.relu(F.max_pool2d(self.bn1(self.conv1(y)), kernel_size=2, stride=2)) # Nx32x6x6
        y = F.relu(F.max_pool2d(self.bn2(self.conv2(y)), kernel_size=2, stride=2)) # Nx64x2x2
        y = self.dropout_ws(y) # Nx64x2x2 -> some element are put randomly to zero by dropout
        y = F.relu(self.fc1(y.view(-1, 256))) # Nxnb_hidden
        y = self.dropout_ws(y) # Nxnb_hidden -> some element are put randomly to zero by dropout
        y = self.fc2(y) # Nx10
        
        # concatenate layers 
        z = torch.cat([x, y], 1) # Nx20
        
        # FC layer for binary classification 
        z = F.relu(self.fc3(z)) #Nx100
        z = self.dropout_comp(z) # Nx100 -> some element are put randomly to zero by dropout
        z = F.relu(self.fc4(z)) #Nx2
        
        return  z