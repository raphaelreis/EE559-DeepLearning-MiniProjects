import sys
sys.path.append('..')
import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
import torch.utils.data as dt
from torch.utils.data import Dataset, DataLoader

#####################################################################################################
#                               Google_Net inspired network                                         #
#           Use the classical inception bloc of the Google_Net CNN over the two channels            #
#       Use auxiliary loss on the output of the inception bloc that classify the digit number       #
#            Resulting features of the inception over the two channels are concatenated             # 
#                          FC layers over the concatenated result                                   #
#####################################################################################################

class conv_block(nn.Module) :
    """
    
     A basic 2d convolution with batch norm -> readibility of the code
     
    """
    
    def __init__(self, in_channels,out_channels,kernel_size = 1,stride =1, padding = 0) :
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride ,padding)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self,x) :
        x = self.bn(self.conv(x))
        return x

############################################################################################################################
    

class Inception_block(nn.Module):
    """
     General :
      Input : Nx1x14x14
      Output by default : Nx256x14x14 
     
     Inception block with four different filters scales : 
      
      By default :
      1) 1x1 Convolution : Nx1x14x14 -> Nx64x14x14
      2) 3x3 Convolution : Nx1x14x14 -> Nx64x14x14
      3) 5x5 Convolution : Nx1x14x14 -> Nx64x14x14
      4) 3x3 Max pooling : Nx1x14x14 -> Nx64x14x14
      
     Concatenate the channels of each scales by default :
      
      [Nx64x14x14,Nx64x14x14,Nx64x14x14,Nx64x14x14] -> [Nx256x14x14]
     
     Parameters : Not tuned in this project, only use default 64
      
      channels_1x1,channels_3x3,channels_5x5,pool_channels -> ouptut number of channels of each scales 
      
    
    """
    def __init__(self,in_channels,channels_1x1,channels_3x3,channels_5x5,pool_channels):
        
        super(Inception_block, self).__init__()
        
        # 1x1 convolution
        self.conv1x1 = conv_block(in_channels,channels_1x1, kernel_size = 1)
        
        # 3x3 convolution factorized in 1x3 followed by 3x1
        self.conv3x3 = nn.Sequential(conv_block(in_channels,channels_3x3, kernel_size = 1),
                                     conv_block(channels_3x3, channels_3x3, kernel_size = (1,3), padding = (0,1)),
                                     conv_block(channels_3x3, channels_3x3, kernel_size = (3,1), padding = (1,0)))
        
        # 5x5 convolution factorized in two consecutive 3x3 implemented as above
        self.conv5x5 = nn.Sequential(conv_block(in_channels,channels_5x5, kernel_size = 1),
                                     conv_block(channels_5x5, channels_5x5, kernel_size = (1,3),padding =(0,1)),
                                     conv_block(channels_5x5, channels_5x5, kernel_size = (3,1), padding = (1,0)),
                                     conv_block(channels_5x5,channels_5x5, kernel_size = (1,3),padding=(0,1)),
                                     conv_block(channels_5x5, channels_5x5, kernel_size = (3,1),padding = (1,0)))
        
        # pooling layer 
        self.pool = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                                  conv_block(in_channels, pool_channels, kernel_size=1))

        
    def forward(self, x):
        
        # compute the four filter of the inception block :  Nx64x14x14
        scale1 = F.relu(self.conv1x1(x))
        scale2 = F.relu(self.conv3x3(x))
        scale3 = F.relu(self.conv5x5(x))
        scale4 = F.relu(self.pool(x))
        
        # concatenate layer for next result
        outputs = [scale1, scale2, scale3, scale4]
        
        # Nx256x14x14
        filters_cat = torch.cat(outputs,1)
        
        return filters_cat

############################################################################################################################

class Auxiliary_loss (nn.Module) :
    """
    General : CNN which output is used for digit recognition 0-9 -> auxiliary loss
    
     Input by default : Nx256x14x14 -> 256 by default refer to inception bloc 
     Output :  Nx10
     
     Parameters :
      
      1) in_channels : number of channels of the ouput tensor of the inception bloc  ->  default 256
      2) drop_prob_aux : dropout rate of the drpout modeule used in the FC layers
      
    """
    
    def __init__(self,in_channels,drop_prob_aux):
        super(Auxiliary_loss, self).__init__()
        
        # Convolutional layers
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        
        # FC layers
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
        #dropout
        self.dropout= nn.Dropout(drop_prob_aux)

    def forward(self, x):
  
        x = F.adaptive_avg_pool2d(x, (4, 4)) #N x 256 x 14 x 14
        x = self.conv(x) # N x 128 x 4 x 4
        x = x.view(-1,2048) # N x 2048
        x = self.dropout(x) # N x 2048 -> some element are put randomly to zero by dropout
        x = F.relu(self.fc1(x)) # N x 1024
        x = self.dropout(x) # N x 1024 -> some element are put randomly to zero by dropout
        x = self.fc2(x) # N x 10

        return x
    
##########################################################################################################################
    
class Google_Net (nn.Module) :
    """
    
    General : Google Net inspired network
        
        Input : Nx2x14x14 -> N pairs of digits (14x14) in a two-channels image
        Output : Nx10, Nx10 -> tensor of two features for digit classification, Nx2 -> N tensor of two features for binary classification
        
        Use inception block over each of the two channels 
        Use CNN for auxiliary loss on the output of the inception bloc
        Concatenate the result of the above CNN for binary classification -> first digit lesser or equal to the second
        
        Parameters :
        
         1) channels_1x1,channels_3x3,channels_5x5,pool_channels -> ouptut number of channels of each filter scales of the inception bloc
         2) nhidden : number of nodes of the second FC layers
         3) drop_prob_comp : dropout rate of the dropout module used in the FC layers for binary classification
         4) drop_prob_aux : dropout rate of Auxiliary CNN
        
    """
    
    def __init__(self,channels_1x1 = 64,channels_3x3 = 64,channels_5x5 =64,pool_channels = 64,nhidden = 60,
                 drop_prob_comp = 0,drop_prob_aux = 0.7):
        super(Google_Net, self).__init__()
        
        # inception block
        self.inception = Inception_block(1,channels_1x1,channels_3x3,channels_5x5,pool_channels)
        
        # Auxiliary CNN
        self.auxiliary = Auxiliary_loss(256,drop_prob_aux)
        
        # FC layers for binary classification
        self.fc1 = nn.Linear(20, nhidden)
        self.fc2 = nn.Linear(nhidden, 100)
        self.fc3 = nn.Linear(100, 2)
        
        # dropout
        self.dropout_comp = nn.Dropout(drop_prob_comp)
        
    def forward(self, input_):
        
        # split the 2-channel input into two 1*14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # inception blocks
        x = self.inception(x) #Nx256x14x14
        y = self.inception(y) #Nx256x14x14
        
        
        # auxiliary loss 
        x = self.auxiliary(x) #Nx10
        y = self.auxiliary(y) #Nx10
        
        # concatenate layers  
        z = torch.cat([x, y], 1) #Nx20
        
        z = F.relu(self.fc1(z)) # Nxnhidden
        z = F.relu(self.fc2(z)) # Nx100
        z = self.dropout_comp(z) #Nx100 -> some element are put randomly to zero by dropout
        z = self.fc3(z) # Nx2
        
        
        return x,y,z