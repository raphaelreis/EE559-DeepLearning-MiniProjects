import torch 
from torch import nn 
from torch.nn import functional as F
from torch import optim
from utils.evaluate import compute_metrics
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

##################################################
#         Le_Net inspired network                #
##################################################

class LeNet(nn.Module):
    """ 
    
    """
    def __init__(self, nb_hidden):
        super(LeNet_WS_sequential, self).__init__()
        
        # convolutional weights for digit reocgnition shared for each image
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
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
        y = F.relu(F.max_pool2d(self.conv3(y), kernel_size=2, stride=2))
        y = F.relu(F.max_pool2d(self.conv4(y), kernel_size=2, stride=2))
        y = F.relu(self.fc1(y.view(-1, 256)))
        y = self.fc2(y)
        
        # concatenate layers 
        z = torch.cat([x, y], 1)
        
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        
        return  z

###################################################################################################################

##### train function ######

def train_Le_Net (model, train_data, validation_data, mini_batch_size=100, optimizer = optim.SGD,
                criterion = nn.CrossEntropyLoss(), n_epochs=40, eta=1e-1,lambda_l2=0, alpha=0.5, beta=0.5):
    
    """
    Train network with auxiliary loss + weight sharing and record train/validation history
    
    """
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    optimizer = optimizer(model.parameters(), lr = eta, weight_decay = lambda_l2)
    
    train_loader = DataLoader(train_data, batch_size=mini_batch_size, shuffle=True)
    
    for e in range(n_epochs):
        epoch_loss = 0
        model.train(True)
        for i, data in enumerate(train_loader, 0):
            
            input_, target_, classes_ = data
            class_1, class_2, out = model(input_)
            aux_loss1 = criterion(class_1, classes_[:,0])
            aux_loss2 = criterion(class_2, classes_[:,1])
            out_loss  = criterion(out, target_)
            net_loss = (alpha * (out_loss) + beta * (aux_loss1 + aux_loss2) )
            epoch_loss += net_loss
            
            optimizer.zero_grad()
            net_loss.backward()
            optimizer.step()
            
        tr_loss, tr_acc = compute_metrics(model, train_data)
        val_loss, val_acc = compute_metrics(model, validation_data)
        
        train_losses.append(tr_loss)
        train_acc.append(tr_acc)
        valid_acc.append(val_acc)
        valid_losses.append(val_loss)
            
        print('Train Epoch: {}  | Loss {:.6f}'.format(
                e, epoch_loss.item()))
        
    return train_losses, train_acc, valid_losses, valid_acc