import sys
sys.path.append('..')
import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
import torch.utils.data as dt
from torch.utils.data import Dataset, DataLoader
from utils.evaluate_aux import compute_metrics

###################################################
#        Google_Net inspired network              #
#    Use weight sharing over the two channels     #
#Use auxiliary loss that classify the digit number#
###################################################

class conv_block(nn.Module) :
    """
    basic 2d convolution with batch norm
    
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
    Inception block with four different filters scale
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
        filter_cat = torch.cat(outputs,1)
        
        return filter_cat

############################################################################################################################

class Auxiliary_loss (nn.Module) :
    """
    auxiliary loss classification of the digit number 0-9
    """
    
    def __init__(self,in_channels,drop_prob_aux,nb_classes = 10):
        super(Auxiliary_loss, self).__init__()
        
        # Convolutional layer
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        
        # Linear classifier
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, nb_classes)
        self.dropout= nn.Dropout(drop_prob_aux)

    def forward(self, x):
        #N x 256 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # N x 256 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(-1,2048)
        # N x 2048
        x = F.relu(self.fc1(x))
        # N x 1024
        x = self.dropout(x)
        # N x 10
        x = self.fc2(x)

        return x
    
##########################################################################################################################
    
class Google_Net (nn.Module) :
    """
    Google net implementing two inception layer in parralel for each channel
    Use auxiliary loss to classify the digit number
    Concatenate the number classification feature map and classify the two channel
    """
    
    def __init__(self,channels_1x1 = 64,channels_3x3 = 64,channels_5x5 =64,pool_channels = 64,nhidden = 60,
                 drop_prob = 0,drop_prob_aux = 0.7,nb_classes = 10):
        super(Google_Net, self).__init__()
        
        # local response norm
        #self.conv1 = conv_block(1, 32, kernel_size = 3, padding = (3 - 1)//2)
        
        #inception block
        self.inception = Inception_block(1,channels_1x1,channels_3x3,channels_5x5,pool_channels)
        
        #auxiliary
        self.auxiliary = Auxiliary_loss(256,drop_prob_aux)
        
        # weights for binary classification 
        self.fc1 = nn.Linear(20, nhidden)
        self.fc2 = nn.Linear(nhidden, 90)
        self.fc3 = nn.Linear(90, 2)
        #self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, input_):
        
        # split the 2-channel input into two 1*14*14 images
        x = input_[:, 0, :, :].view(-1, 1, 14, 14)
        y = input_[:, 1, :, :].view(-1, 1, 14, 14)
        
        # Local response norm
        #x = self.conv1(x)
        #y = self.conv1(y)
        
        # inception blocks
        x = self.inception(x)
        y = self.inception(y)
        
        
        # auxiliary loss 
        x = self.auxiliary(x)
        y = self.auxiliary(y)
        
        # concatenate layers  
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        #z = self.dropout(z)
        z = self.fc3(z)
        
        
        return x,y,z
    
#################################################################################################################################
    
def train_inception (model, train_data, validation_data, device, mini_batch_size=100, optimizer = optim.SGD,
                criterion = nn.CrossEntropyLoss(), n_epochs=40, eta=1e-1, alpha=0.5, beta=0.5):
    
    """
    Train network with auxiliary loss + weight sharing and record train/validation history
    
    """
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    model.train()
    optimizer = optimizer(model.parameters(), lr = eta)
    
    train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
    
    for e in range(n_epochs):
        epoch_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            input_, target_, classes_ = data

            input_ = input_.to(device)
            target_ = target_.to(device)
            classes_ = classes_.to(device)

            class_1, class_2, out = model(input_)
            aux_loss1 = criterion(class_1, classes_[:,0])
            aux_loss2 = criterion(class_2, classes_[:,1])
            out_loss  = criterion(out, target_)
            net_loss = (alpha * (out_loss) + beta * (aux_loss1 + aux_loss2) )
            epoch_loss += net_loss
            
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            
        tr_loss, tr_acc = compute_metrics(model, train_data, device)
        val_loss, val_acc = compute_metrics(model, validation_data, device)
        
        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        valid_acc.append(val_acc)
        valid_losses.append(val_loss)
            
        print('Train Epoch: {}  | Loss {:.6f}'.format(
                e, epoch_loss.item()))
        
    return train_losses, train_acc, valid_losses, valid_acc